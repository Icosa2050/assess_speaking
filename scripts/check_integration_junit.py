"""Require executed, passing mandatory integration cases; collection/skips cannot pass."""
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def mandatory_cases(lane):
    if lane == "sample":
        cases = json.loads((Path(__file__).resolve().parents[1] / "tests/fixtures/asr/sample_references.json").read_text())["cases"]
        return {"test_reference_manifest_matches_tracked_audio", "test_silence_does_not_invent_speech"} | {
            f"test_sample_transcription_content_metrics_and_timestamps[{c['language']}-{c['goal']}]" for c in cases
        } | {f"test_longer_m4a_decoding_and_word_timing[{name}]" for name in ("test1.m4a", "test2.m4a")} | {
            f"test_moderate_seeded_noise_preserves_sample_content[{language}]" for language in ("en", "it")
        }
    if lane == "cloud":
        return {f"test_real_cloud_pipeline[{name}]" for name in ("free", "paid-fallback", "disabled", "auth", "schema", "unknown", "resume", "chatgpt")} | {"test_groq_language_uncertainty_withholds_grade_and_positive_coaching"}
    if lane == "recovery":
        return {"test_unknown_paid_result_survives_restart_and_budget_blocks_post"} | {
            f"test_published_reply_survives_process_loss_and_resume[{loss}]" for loss in ("worker-loss", "backend-loss")
        } | {f"test_cross_process_ledger_and_prompt_serialization[{kind}]" for kind in ("ledger", "cache")}
    if lane == "codec":
        return {"test_real_upload_cap_split_and_overlap_only_speech"}
    raise ValueError("Unknown integration lane")


def verify(lane, path):
    expected = mandatory_cases(lane)
    nodes = list(ET.parse(path).iter("testcase"))
    passed = {node.attrib["name"] for node in nodes if not any(node.find(kind) is not None for kind in ("failure", "error", "skipped"))}
    counts = Counter(node.attrib["name"] for node in nodes)
    missing = expected - passed | {name for name in expected if counts[name] != 1}
    if missing:
        raise ValueError("Mandatory cases did not pass: " + ", ".join(sorted(missing)))
    print(f"{lane}: {len(expected)} mandatory cases executed and passed")


if __name__ == "__main__":
    verify(sys.argv[1], sys.argv[2])
