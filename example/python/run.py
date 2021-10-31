import argparse
import time
from typing import Optional

import core
import soundfile

from forwarder import Forwarder


def run(
    model_dir: str,
    use_gpu: bool,
    text: str,
    speaker_id: int,
    f0_speaker_id: Optional[int],
    f0_correct: float,
    trial_count: int,
) -> None:
    # コアの初期化
    start = time.time()
    core.initialize(model_dir, use_gpu)
    print(f"initialize {time.time() - start: .2f}s")

    # 音声合成処理モジュールの初期化
    start = time.time()
    forwarder = Forwarder(
        yukarin_s_forwarder=core.yukarin_s_forward,
        yukarin_sa_forwarder=core.yukarin_sa_forward,
        decode_forwarder=core.decode_forward,
    )
    print(f"forwarder {time.time() - start: .2f}s")

    # 音声合成
    for trial in range(trial_count):
        start = time.time()
        wave = forwarder.forward(
            text=text + (str(trial) if trial else ""),
            speaker_id=speaker_id,
            f0_speaker_id=f0_speaker_id if f0_speaker_id is not None else speaker_id,
            f0_correct=f0_correct,
        )
        print(f"forward {time.time() - start: .2f}s")

    # 保存
    soundfile.write(f"{text}-{speaker_id}.wav", data=wave, samplerate=24000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker_id", type=int, required=True)
    parser.add_argument("--f0_speaker_id", type=int)
    parser.add_argument("--f0_correct", type=float, default=0)
    parser.add_argument("--trial_count", type=int, default=1)
    run(**vars(parser.parse_args()))
