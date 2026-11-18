"""
predict_student.py - Generate predictions for student_test (no ground truth).
Saves arc-ID route files to predictions/student_<id>.route
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_matching import process_scenario

base_dir    = os.path.dirname(os.path.abspath(__file__))
workspace   = os.path.dirname(base_dir)
student_dir = os.path.join(workspace, 'Student folder', 'Student folder', 'student_test')
out_dir     = os.path.join(base_dir, 'predictions')
os.makedirs(out_dir, exist_ok=True)

scenarios = sorted([d for d in os.listdir(student_dir)
                    if os.path.isdir(os.path.join(student_dir, d))])

print("="*52)
print("PathSnap - Student Test Predictions")
print("="*52)

for sid in scenarios:
    folder = os.path.join(student_dir, sid)
    t0 = time.time()
    pred, _ = process_scenario(folder, sid, has_ground_truth=False)
    dt = time.time() - t0
    out = os.path.join(out_dir, f'student_{sid}.route')
    with open(out, 'w') as f:
        f.write('\n'.join(map(str, pred))+'\n')
    print(f"  {sid}: {len(pred)} arcs  {dt:.1f}s  -> {os.path.basename(out)}")

print(f"\nSaved to: {out_dir}")
