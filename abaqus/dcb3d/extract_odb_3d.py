"""Extract the tip opening / total reaction history from dcb3d.odb into a CSV for
compare_abaqus_3d.py. The tip is a node LINE in 3D, so the reaction is summed over all
tip-top nodes (and the opening averaged); the node labels come from dcb3d_meta.csv written
by generate_inp_3d.py.

Must be run with Abaqus's own embedded Python (odbAccess), from this directory:
    & "C:\\SIMULIA\\Commands\\abaqus.bat" python extract_odb_3d.py
"""
import csv
import sys

from odbAccess import openOdb


def read_meta(path):
    labels = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            labels[row["set"]] = [int(t) for t in row["labels"].split()]
    return labels["tip_top"], labels["tip_bottom"]


def node_series(step, label, var):
    region = step.historyRegions["Node PART-1-1.%d" % label]
    return dict(region.historyOutputs[var].data)


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "dcb3d"
    top_labels, bottom_labels = read_meta("dcb3d_meta.csv")
    odb = openOdb(job + ".odb", readOnly=True)
    step = odb.steps[list(odb.steps.keys())[0]]

    u_top = [node_series(step, n, "U2") for n in top_labels]
    rf_top = [node_series(step, n, "RF2") for n in top_labels]
    u_bot = [node_series(step, n, "U2") for n in bottom_labels]
    rf_bot = [node_series(step, n, "RF2") for n in bottom_labels]

    times = sorted(u_top[0].keys())
    rows = []
    for t in times:
        delta = (sum(d[t] for d in u_top) / len(u_top)
                 - sum(d[t] for d in u_bot) / len(u_bot))
        P = 0.5 * (sum(d[t] for d in rf_top) - sum(d[t] for d in rf_bot))
        rows.append((t, delta, P))

    out = job + "_result.csv"
    with open(out, "w") as f:
        f.write("time,delta,P\n")
        for r in rows:
            f.write("%.8g,%.8g,%.8g\n" % r)

    odb.close()
    print("Wrote %s (%d rows)" % (out, len(rows)))


if __name__ == "__main__":
    main()
