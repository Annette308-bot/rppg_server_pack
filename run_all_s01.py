import sys, subprocess, json

S = "S01"
METHOD = "thesis_precomputed"   # change to "ceemdan_hilbert" if you want

clips = [
    ("rest", "face",   r"..\Data\raw\my_phone\S01\face\S01_rest_face.mp4"),
    ("rest", "palm",   r"..\Data\raw\my_phone\S01\palm\S01_rest_palm.mp4"),
    ("exercise", "face", r"..\Data\raw\my_phone\S01\face\S01_exercise_face.mp4"),
    ("exercise", "palm", r"..\Data\raw\my_phone\S01\palm\S01_exercise_palm.mp4"),
    ("breath", "face", r"..\Data\raw\my_phone\S01\face\S01_breath_face.mp4"),
    ("breath", "palm", r"..\Data\raw\my_phone\S01\palm\S01_breath_palm.mp4"),
]

print("condition\tmodality\thr_bpm\tvalid_pct\ttrusted\tselected_imf\tiqr")

for cond, mod, v in clips:
    p = subprocess.run(
        [
            sys.executable, "realworld_demo_rppg_single.py",
            "--video", v,
            "--subject", S,
            "--condition", cond,
            "--modality", mod,
            "--outdir", r"outputs\server_results",
            "--method", METHOD,
        ],
        capture_output=True,
        text=True,
    )

    out = p.stdout.strip().splitlines()
    if not out:
        print(f"{cond}\t{mod}\tERROR(no output)\t-\t-\t-\t-")
        continue

    d = json.loads(out[-1])
    print(
        f"{cond}\t{mod}\t{d.get('hr_bpm')}\t{d.get('valid_pct_masked')}\t"
        f"{d.get('trusted')}\t{d.get('selected_imf')}\t{d.get('iqr_bpm')}"
    )
