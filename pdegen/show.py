"""Render a checked wire record for review in plain text."""
import argparse
import json
from collections import Counter
from pathlib import Path
from .contract import validate


def expression(terms):
    parts = []
    for monomial, coefficient in terms.items():
        variables = [] if monomial=="1" else monomial.split("*")
        factors = [v if n==1 else f"{v}^{n}" for v,n in Counter(variables).items()]
        value = coefficient.lstrip("-")
        if value!="1" or not factors:
            factors.insert(0,value)
        parts.append(("-" if coefficient.startswith("-") else "+", "*".join(factors)))
    return " ".join(f"{s} {v}" for s,v in parts).lstrip("+ ").replace("- ","-") or "0"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("record",type=Path)
    item=validate(json.loads(p.parse_args().record.read_text()))
    print(item["instance_id"],item["parameters"])
    eq=item["equations"]
    for i,name in enumerate(item["variables"]):
        D,A,F,G = (expression(eq[k][i]) for k in ("D","A","F","G"))
        print(f"{name}_t = ({D}) {name}_xx - ({A}) {name}_x + ({F}) + dx({G})")
    print(item["target_wave"]["basis"])
    print("U = ("+", ".join(map(expression,item["target_wave"]["profile"]))+")")
    print("c =",item["target_wave"]["speed"]," / ",item["target_wave"]["phase"])
    for key in ("B","C"):
        print(key)
        for row in item["linearised_operator"][key]:
            print("  ["+", ".join(map(expression,row))+"]")
    print(item["function_space"],item["operator_domain"],item["stability_question"])


if __name__=="__main__":
    main()
