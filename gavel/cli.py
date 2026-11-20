"""Command line interface for scoring and benchmarking."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Callable

from .backends.openai_compat import OpenAICompatBackend
from .judge import check_single_token_labels, judge
from .metrics import pearson, recall_fp, run_noise, spearman
from .rubric import load_rubric


BackendFactory = Callable[[str, str, str | None], object]


def _add_shared(parser: argparse.ArgumentParser, *, defaults: bool) -> None:
    default = None if defaults else argparse.SUPPRESS
    parser.add_argument(
        "--backend",
        choices=("openai", "native", "completions"),
        default=("openai" if defaults else default),
    )
    parser.add_argument("--tokenizer", default=default)
    parser.add_argument("--revision", default=default)
    parser.add_argument("--prompt-format", choices=("chatml",), default=default)
    parser.add_argument("--template-kwargs", default=("{}" if defaults else default))
    parser.add_argument("--model-path", default=default)
    parser.add_argument("--device", default=("cuda" if defaults else default))
    parser.add_argument("--dtype", default=("bfloat16" if defaults else default))
    parser.add_argument("--chunk", type=int, default=(256 if defaults else default))
    parser.add_argument(
        "--base-url",
        default=(os.environ.get("GAVEL_BASE_URL", "http://localhost:8000") if defaults else default),
    )
    parser.add_argument(
        "--model", default=(os.environ.get("GAVEL_MODEL", "default") if defaults else default)
    )
    parser.add_argument(
        "--api-key", default=(os.environ.get("GAVEL_API_KEY") if defaults else default)
    )
    parser.add_argument("--mode", choices=("lean", "explain"), default=("lean" if defaults else default))
    parser.add_argument("--concurrency", type=int, default=(8 if defaults else default))
    parser.add_argument(
        "--no-label-check", action="store_true", default=(False if defaults else default)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gavel")
    _add_shared(parser, defaults=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score")
    _add_shared(score, defaults=False)
    score.add_argument("rubric")
    score.add_argument("items")
    score.add_argument("--out")

    bench = subparsers.add_parser("bench")
    _add_shared(bench, defaults=False)
    bench.add_argument("rubric")
    bench.add_argument("labelled")
    bench.add_argument("--runs", type=int, default=2)
    bench.add_argument("--json", action="store_true")

    serve = subparsers.add_parser("serve")
    _add_shared(serve, defaults=False)
    serve.add_argument("--rubric", action="append", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8090)
    return parser


def _load_jsonl(path: str, *, labelled: bool) -> list[dict]:
    items = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(item, dict) or "id" not in item or not isinstance(item.get("text"), str):
                raise ValueError(f"{path}:{line_number}: expected id and text")
            if labelled and (
                not isinstance(item.get("label"), (int, float))
                or isinstance(item.get("label"), bool)
            ):
                raise ValueError(f"{path}:{line_number}: expected numeric label")
            items.append(item)
    return items


def _tokens_per_item(tokens: int, item_count: int) -> float:
    return tokens / item_count if item_count else 0.0


def _default_backend(args: argparse.Namespace) -> object:
    if args.backend == "openai":
        return OpenAICompatBackend(args.base_url, args.model, api_key=args.api_key)
    if args.backend == "completions":
        from .backends.completions import (
            CompletionsBackend,
            chatml_renderer,
            tokenizer_renderer,
        )

        render = (
            tokenizer_renderer(
                args.tokenizer,
                args.template_kwargs,
                revision=args.revision,
            )
            if args.tokenizer
            else chatml_renderer
        )
        return CompletionsBackend(
            args.base_url,
            args.model,
            api_key=args.api_key,
            render=render,
        )

    from .backends.native import NativeBackend

    return NativeBackend.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        chunk=args.chunk,
        revision=args.revision,
    )


def _native_stats(backend: object) -> dict[str, int | float] | None:
    stats = getattr(backend, "stats", None)
    if stats is None or not hasattr(stats, "prefill_tokens"):
        return None
    return {
        "prefill_tokens": stats.prefill_tokens,
        "reused_tokens": stats.reused_tokens,
        "prefill_seconds": stats.prefill_s,
        "decode_tokens": stats.decode_tokens,
        "decode_seconds": stats.decode_s,
    }


def _native_stats_delta(
    before: dict[str, int | float] | None,
    after: dict[str, int | float] | None,
) -> dict[str, int | float] | None:
    if before is None or after is None:
        return None
    delta = {key: after[key] - before[key] for key in before}
    seconds = float(delta["prefill_seconds"])
    delta["prefill_tok_per_s"] = (
        float(delta["prefill_tokens"]) / seconds if seconds else 0.0
    )
    return delta


def _native_stats_line(stats: dict[str, int | float]) -> str:
    return (
        f"prefill_tokens {stats['prefill_tokens']} "
        f"reused_tokens {stats['reused_tokens']} "
        f"prefill_seconds {stats['prefill_seconds']:.3f} "
        f"prefill_tok_per_s {stats['prefill_tok_per_s']:.1f} "
        f"decode_tokens {stats['decode_tokens']} "
        f"decode_seconds {stats['decode_seconds']:.3f}"
    )


async def _close_backend(backend: object) -> None:
    close = getattr(backend, "aclose", None)
    if close is not None:
        result = close()
        if inspect.isawaitable(result):
            await result


async def _score(args: argparse.Namespace, backend: object, rubric, items: list[dict]) -> int:
    if not args.no_label_check:
        await check_single_token_labels(backend, rubric)
    results, stats = await judge(
        backend, rubric, items, mode=args.mode, concurrency=args.concurrency
    )
    lines = [json.dumps(result, ensure_ascii=False) for result in results]
    if args.out:
        Path(args.out).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    else:
        for line in lines:
            print(line)
    errors = sum("error" in result for result in results)
    print(
        f"items {len(items)} ok {len(items) - errors} errors {errors} "
        f"calls {stats.calls} generated_tokens_per_item "
        f"{_tokens_per_item(stats.generated_tokens, len(items)):.3f} "
        f"wall_seconds {stats.wall_s:.3f}",
        file=sys.stderr,
    )
    native = _native_stats(backend)
    if native is not None:
        native["prefill_tok_per_s"] = (
            float(native["prefill_tokens"]) / float(native["prefill_seconds"])
            if native["prefill_seconds"]
            else 0.0
        )
        print(_native_stats_line(native), file=sys.stderr)
    return 1 if errors else 0


async def _bench(args: argparse.Namespace, backend: object, rubric, items: list[dict]) -> int:
    if args.runs < 1:
        raise ValueError("runs must be at least 1")
    if not args.no_label_check:
        await check_single_token_labels(backend, rubric)
    truth_by_id = {str(item["id"]): float(item["label"]) for item in items}
    reports = []
    score_maps = []
    for run_number in range(1, args.runs + 1):
        native_before = _native_stats(backend)
        results, stats = await judge(
            backend, rubric, items, mode=args.mode, concurrency=args.concurrency
        )
        native = _native_stats_delta(native_before, _native_stats(backend))
        scores = {
            str(result["id"]): float(result["score"])
            for result in results
            if "error" not in result and result.get("score") is not None
        }
        ids = [str(item["id"]) for item in items if str(item["id"]) in scores]
        truth = [truth_by_id[item_id] for item_id in ids]
        predicted = [scores[item_id] for item_id in ids]
        report = {
            "run": run_number,
            "n": len(scores),
            "spearman": spearman(truth, predicted),
            "pearson": pearson(truth, predicted),
            "generated_tokens_per_item": _tokens_per_item(stats.generated_tokens, len(items)),
            "wall_seconds": stats.wall_s,
        }
        if rubric.threshold is not None:
            report["threshold"] = rubric.threshold
            report.update(recall_fp(truth, predicted, rubric.threshold))
        if native is not None:
            report["native"] = native
        reports.append(report)
        score_maps.append(scores)
    output = {"runs": reports}
    if args.runs >= 2:
        output["noise"] = run_noise(score_maps[0], score_maps[1])
    if args.json:
        print(json.dumps(output, ensure_ascii=False))
    else:
        for report in reports:
            fields = [
                f"run {report['run']}",
                f"n {report['n']}",
                f"spearman {report['spearman']}",
                f"pearson {report['pearson']}",
            ]
            if rubric.threshold is not None:
                fields.extend(
                    [
                        f"recall {report['recall']}",
                        f"fp {report['fp']}",
                    ]
                )
            fields.extend(
                [
                    f"generated_tokens_per_item {report['generated_tokens_per_item']:.3f}",
                    f"wall_seconds {report['wall_seconds']:.3f}",
                ]
            )
            print(" ".join(fields))
            if "native" in report:
                print(_native_stats_line(report["native"]))
        if "noise" in output:
            noise = output["noise"]
            print(
                f"noise n {noise['n']} mean_abs_delta {noise['mean_abs_delta']} "
                f"max_abs_delta {noise['max_abs_delta']} moved_ge_1 {noise['moved_ge_1']}"
            )
    return 0


async def _run(args: argparse.Namespace, backend_factory: BackendFactory | None) -> int:
    try:
        rubric = load_rubric(args.rubric)
        source = args.items if args.command == "score" else args.labelled
        items = _load_jsonl(source, labelled=args.command == "bench")
    except (OSError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    if backend_factory is None:
        backend = _default_backend(args)
    else:
        backend = backend_factory(args.base_url, args.model, args.api_key)
    try:
        if args.command == "score":
            return await _score(args, backend, rubric, items)
        return await _bench(args, backend, rubric, items)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        await _close_backend(backend)


def _serve(args: argparse.Namespace, backend_factory: BackendFactory | None) -> int:
    try:
        import uvicorn

        from .server import create_app
    except ImportError:
        print(
            'gavel serve needs the server extra: pip install "thegavel[server]"',
            file=sys.stderr,
        )
        return 2

    try:
        rubric_list = [load_rubric(path) for path in args.rubric]
    except (OSError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    if backend_factory is None:
        backend = lambda: _default_backend(args)
    else:
        backend = lambda: backend_factory(args.base_url, args.model, args.api_key)
    app = create_app(
        backend,
        {rubric.name: rubric for rubric in rubric_list},
        label_check=not args.no_label_check,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None, backend_factory: BackendFactory | None = None) -> int:
    """Run the command line interface and return its process status."""

    args = _parser().parse_args(argv)
    if args.backend == "native" and not args.model_path:
        print(
            "input error: --model-path is required for --backend native",
            file=sys.stderr,
        )
        return 2
    if args.backend == "completions":
        if bool(args.tokenizer) == bool(args.prompt_format):
            print(
                "input error: --backend completions needs --tokenizer or --prompt-format",
                file=sys.stderr,
            )
            return 2
        try:
            args.template_kwargs = json.loads(args.template_kwargs)
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"input error: invalid --template-kwargs JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(args.template_kwargs, dict):
            print("input error: --template-kwargs must be a JSON object", file=sys.stderr)
            return 2
    if args.command == "serve":
        return _serve(args, backend_factory)
    return asyncio.run(_run(args, backend_factory))
