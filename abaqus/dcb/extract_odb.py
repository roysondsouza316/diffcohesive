"""Extract the tip opening displacement / reaction-force history from dcb_ref.odb into a CSV,
for comparison against diffcohesive's own load-displacement curve (see compare_abaqus.py).

Must be run with Abaqus's own embedded Python (odbAccess is not installable in a normal
environment), not this repo's conda env:
    & "C:\\SIMULIA\\Commands\\abaqus.bat" python extract_odb.py
"""
import csv

from odbAccess import openOdb

# Node labels (1-based) for the tip nodes, matching generate_inp.py / build_double_cantilever_mesh
# defaults: tip_bottom = node id 0 -> label 1; tip_top = node id 2*ny*(nx+1) = 248 -> label 249.
TIP_BOTTOM_REGION = "Node PART-1-1.1"
TIP_TOP_REGION = "Node PART-1-1.249"


def main():
    import sys
    job = sys.argv[1] if len(sys.argv) > 1 else "dcb_ref"
    odb = openOdb(job + ".odb", readOnly=True)
    step = odb.steps[list(odb.steps.keys())[0]]

    top = step.historyRegions[TIP_TOP_REGION]
    bottom = step.historyRegions[TIP_BOTTOM_REGION]

    u_top = dict(top.historyOutputs["U2"].data)
    rf_top = dict(top.historyOutputs["RF2"].data)
    u_bottom = dict(bottom.historyOutputs["U2"].data)
    rf_bottom = dict(bottom.historyOutputs["RF2"].data)

    times = sorted(u_top.keys())
    rows = []
    for t in times:
        delta = u_top[t] - u_bottom[t]
        P = 0.5 * (rf_top[t] - rf_bottom[t])
        rows.append((t, delta, P))

    out = "dcb_abaqus_result.csv" if job == "dcb_ref" else job + "_result.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "delta", "P"])
        writer.writerows(rows)

    odb.close()
    print("Wrote %s (%d rows)" % (out, len(rows)))


if __name__ == "__main__":
    main()
