import argparse
import torch
from transformers import T5Tokenizer
from torch.utils.data import DataLoader, SequentialSampler
import src.slurm
import src.data
import src.model
from time import time


def evaluate(model, dataset, dataloader, tokenizer, opt):
    model.eval()
    if hasattr(model, "module"):
        model = model.module

    model.overwrite_forward_crossattention()
    model.reset_score_storage()
    f_w = open(opt.write_path, 'w')
    f_w.write(f'qid::q_text::pid::golden_answer::predict_answer::original_score::attention_score'+'\n')
    t1 = time()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask, qid, q_txt) = batch
            # context_ids: <question: > + 'question' + <title: > + 'title' + <passage: > + 'passage'

            model.reset_score_storage()

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50,
            )
            crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]

                for j in range(context_ids.size(1)):
                    f_w.write(qid[k] + '::' + q_txt[k] + '::' + str(example['ctxs'][j]['id'])
                              + '::' + str(example['answers']) + '::' + ans + '::' + str(example['ctxs'][j]['score']) + '::'
                              + str(crossattention_scores[k, j].item()+7) + '\n'
                              )
            if i % 10 == 0:
                print(f'{(i+1)*opt.per_gpu_batch_size} queries processing finished!!')
    f_w.close()
    t2 = time()
    print(f'Total cost time: {t2-t1}')


if __name__ == "__main__":
    print('------------------------------------------- Start --------------------------------------------------------')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # TODO modify following options
    parser.add_argument("--per_gpu_batch_size", default=1, type=int, help="  # 64 for A100(80G) ~5h, 32 for V100(32G) ~18h")
    parser.add_argument('--eval_data', type=str, help='path of retrieval data', default="")
    parser.add_argument('--write_path', type=str, help='path of writing atn score file', default="")
    parser.add_argument('--wiki_path', type=str, help='path of wikipedia corpus', default="")
    parser.add_argument('--reader_path', type=str, help='path of reader checkpoint', default="")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)")
    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
    parser.add_argument('--text_maxlength', type=int, default=200, help='maximum number of tokens in text segments (question+passage)')
    parser.add_argument('--n_context', type=int, default=20)

    opt = parser.parse_args()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    tokenizer = T5Tokenizer.from_pretrained('t5-large', return_dict=False) 
    model_class = src.model.FiDT5 
    model = model_class.from_pretrained(opt.reader_path)
    model = model.to(opt.device)

    collator_function = src.data.Collator_(opt.text_maxlength, tokenizer)
    eval_examples = src.data.load_data_(
        opt.wiki_path,
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset_(
        eval_examples,
        opt.n_context,
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=opt.per_gpu_batch_size,
        num_workers=20,
        collate_fn=collator_function,
        drop_last=False
    )

    evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)


