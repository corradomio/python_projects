import sys
import traceback
from sktimex.forecasting import create_forecaster
from stdlib import jsonx, create_from, qualified_type
from synth import *



def selected(name, included: list[str], excluded: list[str]) -> bool:
    if len(included) == 0 and len(excluded) == 0:
        return True

    for m in included:
        if m in name:
            return True
    for m in excluded:
        if m in name:
            return False
    if len(included) == 0 and len(excluded) == 0:
        return True
    elif len(included) > 0:
        return False
    else:
        return True
# end



def check_model(name, dfdict: dict[tuple, pd.DataFrame], jmodel: dict, override=False):
    if name.startswith("#"):
        return

    print("---", name, "---")

    try:
        # model = create_from(jmodel)
        model = create_forecaster(name, jmodel)
        print("... ...", qualified_type(model))

    except Exception as e:
        print(f"ERROR[{name}]:", e)
        traceback.print_exception(*sys.exc_info())
# end



def check_models(df: pd.DataFrame, jmodels: dict[str, dict], override=False, includes=[], excludes=[]):
    dfdict = pdx.groups_split(df, groups=["cat"])

    for name in jmodels:
        if selected(name, includes, excludes):
            check_model(name, dfdict, jmodels[name], override)



def main():
    print("dataframe")
    df = create_syntethic_data(12 * 8, 0.0, 1, 0.33)

    jmodels = jsonx.load("config/ext_models.json")
    check_models(df, jmodels, override=True)


if __name__ == "__main__":
    main()
