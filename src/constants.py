import os

##############################################################
### directories ##############################################

# TODO: set DIR_PROJECT as location for all data and models
#DIR_PROJECT = '/your/project/directory/here/'
DIR_PROJECT = '/data/dvv/radadapt-test/'
assert os.path.exists(DIR_PROJECT), 'please enter valid directory'

# TODO: for best practice, move data to DIR_PROJECT/data/ (outside repo)
#DIR_DATA = os.path.join(DIR_PROJECT, 'data/') # input data
DIR_DATA = os.path.join(DIR_PROJECT, 'xx/') # input data
if not os.path.exists(DIR_DATA):
    DIR_DATA = 'data/'

# directory of tuned models. created automatically
DIR_MODELS_TUNED = os.path.join(DIR_PROJECT, 'models_tuned/') # tuned models

# directory of physionet's pre-trained models (clin-t5, clin-t5-sci)
# download here: https://www.physionet.org/content/clinical-t5/1.0.0/
DIR_MODELS_CLIN = os.path.join(DIR_PROJECT, 'your/downloaded/models/')


##############################################################
### models ###################################################

MODELS = {
    "t5-base": "t5-base",
    "flan-t5-base": "google/flan-t5-base",
    "scifive-base": "razent/SciFive-base-Pubmed_PMC", 
    "clin-t5-sci": os.path.join(DIR_MODELS_CLIN, 'Clinical-T5-Sci'),
    "clin-t5-base": os.path.join(DIR_MODELS_CLIN, 'Clinical-T5-Base'),

    "t5-large": "t5-large",
    "flan-t5-large": "google/flan-t5-large",
    "scifive-large": "razent/SciFive-large-Pubmed_PMC",
    "clin-t5-large": os.path.join(DIR_MODELS_CLIN, 'Clinical-T5-Large'),
}


##############################################################
### hyperparameters ##########################################

METHODS = ['lora', 'prefix_tuning']

MAX_LEN = 512 # max length for input tokens
MAX_NEW_TOKENS = 50  # max length of tokens to generate
PROMPT_LEN = 20 # for prefix tuning

# general training
BATCH_SIZE_BASE = 16  # for base models
BATCH_SIZE_PREFIX_TUNE = 8  # for large models
BATCH_SIZE_LORA = 6 
LR0_PREFIX_TUNE = 1e-2
LR0_LORA = 1e-3
TRN_EPOCHS_PREFIX_TUNE = 10
TRN_EPOCHS_LORA = 5
PATIENCE = 5

# lora hyperparameters
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1


##############################################################
### prompting ################################################

START_PREFIX = 'summarize the radiology report findings into an impression:\n'
ICL_PROMPT = '[ICL_PROMPT]'

# append DEFAULTS to cases[case_id] if key not manually set
# to override default, include param in a cases[case_id]
DEFAULTS = {
    'method': 'discrete',
    'insert_prefix': 'findings:\n',
    'prompt': '\nimpression:',
    'grad_accum_steps': 4, 
    'lr_n_warmup_steps': 100, 
    'lr_schedule': 'linear_decay', 
    'max_new_tokens': 200,
}

cases = { 
    
    # discrete prompting: null, prefix, and in-context examples (1,2,4)
    0: {},
    1: {'prompt': f'\n{ICL_PROMPT}_1\nimpression:'},
    2: {'prompt': f'\n{ICL_PROMPT}_2\nimpression:'},
    4: {'prompt': f'\n{ICL_PROMPT}_4\nimpression:'},
    5: {'start_prefix': START_PREFIX},

    # prefix tuning, lora (case_id >= 100)
    100: {'method': 'prefix_tuning'},
    200: {'method': 'lora'},

}

def set_method_params(cases, case_id, param_str,
                      val_prefix_tune, val_lora):
    ''' set parameters specific to methods 
        if not explicitly specified in cases '''

    if param_str not in cases[case_id].keys():
        if cases[case_id]['method'] == 'prefix_tuning':
            cases[case_id][param_str] = val_prefix_tune 
        elif cases[case_id]['method'] == 'lora':
            cases[case_id][param_str] = val_lora

    return cases

# append DEFAULTS keys to cases[case_id] only if key dne in cases[case_id]
for case_id in cases:
    for key in DEFAULTS:
        if key not in cases[case_id]:
            cases[case_id][key] = DEFAULTS[key]

    cases = set_method_params(cases, case_id, 'batch_size',
                              BATCH_SIZE_PREFIX_TUNE, BATCH_SIZE_LORA)
    cases = set_method_params(cases, case_id, 'trn_epochs',
                              TRN_EPOCHS_PREFIX_TUNE, TRN_EPOCHS_LORA)
    cases = set_method_params(cases, case_id, 'lr0',
                              LR0_PREFIX_TUNE, LR0_LORA)


##############################################################
### misc filenames ###########################################

FN_FINDING = 'finding.csv'
FN_SUM_REF = 'summary_ref.csv' # reference summary
FN_SUM_GEN = 'summary_gen.csv' # generated summary
FN_SUM_GEN_CLEAN = 'summary_gen_clean.csv' # generated summary
FN_METRICS = 'metrics.json' 
FN_METRICS_CLEAN = 'metrics_clean.json' 
DIR_ICL = os.path.join(DIR_PROJECT, 'icl')
EVAL_CXR_DATA = False
