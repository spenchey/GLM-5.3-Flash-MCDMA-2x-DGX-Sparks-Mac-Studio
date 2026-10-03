from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_native_reference_explicitly_selects_dflash2_for_the_poc():
    script = (ROOT / "scripts" / "start-native-portable.sh").read_text()
    assert "--drafter incoai/GLM-5.3-Flash-DFlash2" in script
    assert 'glm.mcdma.drafter=dflash2' in script
    assert '--mtp-drafts "$THREE_MACHINE_MTP_DRAFTS"' in script
    assert "--no-drafts" not in script
