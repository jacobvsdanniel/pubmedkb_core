import os

def official_f1(eval_dir):
    # Run the perl script
    try:
        cmd = f"perl {eval_dir}/scorer.pl {eval_dir}/proposed_answers.txt {eval_dir}/answer_keys.txt > {eval_dir}/result.txt"
        os.system(cmd)
    except:
        raise Exception("perl is not installed or proposed_answers.txt is missing")

    with open(os.path.join(eval_dir, "result.txt"), "r", encoding="utf-8") as f:
        # macro_result = list(f)[-1]
        # macro_result = macro_result.split(":")[1].replace(">>>", "").strip()
        # macro_result = macro_result.split("=")[1].strip().replace("%", "")
        # macro_result = float(macro_result) / 100
        micro_result = list(f)[-8]
        assert micro_result[-2] == "%"
        micro_result = micro_result[-7:-2]
        micro_result = float(micro_result) / 100

    return micro_result


if __name__ == "__main__":
    print("macro-averaged F1 = {}%".format(official_f1() * 100))
