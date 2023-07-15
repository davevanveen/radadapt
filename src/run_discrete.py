import time
from tqdm import tqdm
import transformers

import constants
import process
import parser


def main():
    # parse arguments. set data paths based on expmt params
    args = parser.get_parser()

    generator = transformers.pipeline(
        "text2text-generation", model=constants.MODELS[args.model]
    )

    # preprocess data. list order will be different in dataloader
    list_finding_ = process.load_data(args, task="test")

    # reference findings + summaries and generated summaries
    list_finding, list_sum_ref, list_sum_gen, idcs = [], [], [], []

    # generate summary for each finding
    t0 = time.time()
    for finding in tqdm(list_finding_, total=len(list_finding_)):
        generated = generator(
            finding["sentence"],
            max_length=constants.MAX_NEW_TOKENS,
            clean_up_tokenization_spaces=True,
        )
        list_sum_gen.append(generated[0]["generated_text"])
        list_sum_ref.append(finding["text_label"])
        list_finding.append(finding["sentence"])
        idcs.append(finding["idx"])  # track idcs to preserve order

    print(
        "generated {} samples for {} expmt in {} sec".format(
            len(list_finding_), args.expmt_name, time.time() - t0)
    )

    process.postprocess_and_save(args, idcs, list_finding,
                                 list_sum_ref, list_sum_gen)


if __name__ == "__main__":
    main()
