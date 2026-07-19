import numpy as np
import pytest

from DecoderDSP import DecoderDSP
from EncoderDSP import EncoderDSP
from Settings import Settings
from WaveParams import WaveParams


def _custom_params(settings: Settings) -> WaveParams:
    params = WaveParams.harmonic_default(settings)
    for i in range(settings.data_offset,
                   settings.data_offset + settings.data_harmonics):
        params.omegas[i] = (i + 1) + 0.5
    params.amps[1] = 0.35
    params.phases[2] = 1.2
    return params


def test_json_round_trip(tmp_path):
    settings = Settings()
    params = _custom_params(settings)
    file_path = str(tmp_path / "wave.json")
    params.to_json_file(file_path)
    loaded = WaveParams.from_json_file(file_path)
    assert loaded.amps == params.amps
    assert loaded.phases == params.phases
    assert loaded.omegas == params.omegas


def test_validate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        WaveParams(amps=[0.5], phases=[0.0, 0.0], omegas=[1.0]).validate()


def test_wave_params_survive_strategy_rebuild():
    enc = EncoderDSP(Settings())
    params = _custom_params(enc.settings)
    enc.set_wave_params(params)
    enc.set_strategy_kind("four")
    assert enc._wave_generator.get_omegas() == params.omegas
    assert enc._wave_generator.get_amps() == params.amps


def test_custom_wave_decodes_only_when_scalars_match(tmp_path):
    payload_path = tmp_path / "payload.txt"
    message = "HELLO ADDITIVE WAVE EDITOR 12345 " * 4
    payload_path.write_text(message)

    def run(matched: bool) -> str:
        enc = EncoderDSP(Settings())
        dec = DecoderDSP(Settings())
        params = _custom_params(enc.settings)
        enc.set_wave_params(params)
        if matched:
            dec.set_harmonic_scalars(params.omegas)
        enc.set_payload_kind("text")
        dec.set_payload_kind("text")
        enc.load_payload_file(str(payload_path))
        enc.set_f0(400.0)
        dec.set_f0(400.0)
        decoded = []
        dec.set_on_raw_text(decoded.append)
        for _ in range(200):
            chunk = enc.process(1024)
            dec.process_chunk(np.array(chunk.get_samples(), dtype=np.float32), 1024)
        return decoded[-1] if decoded else ""

    assert "ADDITIVE WAVE EDITOR" in run(matched=True)
    assert "ADDITIVE WAVE EDITOR" not in run(matched=False)
