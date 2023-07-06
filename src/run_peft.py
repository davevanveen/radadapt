import os
from peft import PeftModel, PeftConfig
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import constants
import parser
import process


def main():

    # parse arguments. set data paths based on expmt params
    args = parser.get_parser()

    # load model, tokenizer
    model = load_model(args) # note: case_id determines training disbn
    tokenizer = AutoTokenizer.from_pretrained(constants.MODELS[args.model])

    # load data
    test_dataset = process.load_data(args, task='test')
    test_loader = process.get_loader(test_dataset, tokenizer, args.batch_size)

    # reference findings + summaries and generated summaries
    list_finding, list_sum_ref, list_sum_gen, idcs = [], [], [], []
    model.eval()

    # generate summary for each finding
    t0 = time.time()
    for step, batch in enumerate(tqdm(test_loader)):
        
        idcs.extend(batch['idx']) # idcs preserve order
        batch = {k: v.to(args.device) for k, v in batch.items()}
        
        with torch.no_grad():

            outputs = model.generate(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     do_sample=False, 
                                     max_new_tokens=args.max_new_tokens)
       
        list_finding.extend(
            tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        )
        list_sum_gen.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

        ref_labels = batch['labels']
        ref_labels[ref_labels == -100] = 0
        list_sum_ref.extend(tokenizer.batch_decode(ref_labels,
                                                   skip_special_tokens=True))

    print('generated {} samples for {} expmt in {} sec'.format(\
          len(list_sum_gen), args.expmt_name, time.time() - t0))

    process.postprocess_and_save(args, idcs, list_finding, list_sum_ref, list_sum_gen)


def load_model(args):
   
    subdirs = [ii[0].split('/')[-1] for ii in os.walk(args.dir_models_tuned)]
    model_epoch = max([int(ii) for ii in subdirs if ii.isdigit()])

    dir_model_peft = os.path.join(args.dir_models_tuned, f'{model_epoch}')
    print(f'evaluating model: {dir_model_peft}')
    config = PeftConfig.from_pretrained(dir_model_peft)
    config.base_model_name_or_path = constants.MODELS[args.model] 
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, dir_model_peft).to(args.device)

    return model


if __name__ == '__main__':
    main()
