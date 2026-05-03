import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model import RelationExtractor


UNK_TOKEN = '<UNK>'
REPO_ROOT = Path(__file__).resolve().parents[2]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    return True


def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    embedding_matrix = []
    for i, (key, entity) in enumerate(embedding_dict.items()):
        entity2idx[key.strip()] = i
        idx2entity[i] = key.strip()
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix


def get_vocab(data):
    word_to_ix = {}
    max_length = 0
    idx2word = {}
    for d in data:
        sent = d[1]
        for word in sent.split():
            if word not in word_to_ix:
                idx2word[len(word_to_ix)] = word
                word_to_ix[word] = len(word_to_ix)

        length = len(sent.split())
        if length > max_length:
            max_length = length

    if UNK_TOKEN not in word_to_ix:
        idx2word[len(word_to_ix)] = UNK_TOKEN
        word_to_ix[UNK_TOKEN] = len(word_to_ix)

    return word_to_ix, idx2word, max_length


def encode_question(question, word2ix):
    unk_idx = word2ix.get(UNK_TOKEN)
    encoded_question = []
    for word in question.strip().split():
        word = word.strip()
        if word in word2ix:
            encoded_question.append(word2ix[word])
        elif unk_idx is not None:
            encoded_question.append(unk_idx)
        else:
            raise KeyError(f"Word '{word}' is not in the vocabulary and {UNK_TOKEN} is missing")
    return encoded_question


def preprocess_entities_relations(entity_dict, relation_dict, entities, relations):
    entity_embeddings = {}
    relation_embeddings = {}

    with open(entity_dict, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            ent_id = int(line[0])
            ent_name = line[1]
            entity_embeddings[ent_name] = entities[ent_id]

    with open(relation_dict, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            rel_id = int(line[0])
            rel_name = line[1]
            relation_embeddings[rel_name] = relations[rel_id]

    return entity_embeddings, relation_embeddings


def in_top_k(scores, ans, k):
    if type(ans) is int:
        ans = [ans]
    topk = torch.topk(scores, k)[1].tolist()
    for x in topk:
        if x in ans:
            return True
    return False


def reciprocal_rank(scores, ans):
    if type(ans) is int:
        ans = [ans]
    ranked = torch.argsort(scores, descending=True)
    best_rank = min((ranked == a).nonzero(as_tuple=True)[0].item() + 1 for a in ans)
    return 1.0 / best_rank


def init_metric_counts():
    return {
        'total': 0,
        'correct': 0,
        'hits_at_1': 0,
        'hits_at_5': 0,
        'hits_at_10': 0,
        'reciprocal_rank_sum': 0.0,
    }


def update_metric_counts(counts, is_correct, hit_at_1, hit_at_5, hit_at_10, reciprocal_rank_value):
    counts['total'] += 1
    counts['correct'] += is_correct
    counts['hits_at_1'] += int(hit_at_1)
    counts['hits_at_5'] += int(hit_at_5)
    counts['hits_at_10'] += int(hit_at_10)
    counts['reciprocal_rank_sum'] += reciprocal_rank_value


def finalize_metric_counts(counts):
    total = counts['total']
    return {
        'Accuracy': counts['correct'] / total,
        'Hits@1': counts['hits_at_1'] / total,
        'Hits@5': counts['hits_at_5'] / total,
        'Hits@10': counts['hits_at_10'] / total,
        'MRR': counts['reciprocal_rank_sum'] / total,
    }


def hop_sort_key(hop):
    try:
        return (0, int(hop))
    except ValueError:
        return (1, hop)


def print_test_metrics(label, metrics):
    print(label)
    print(f'  MRR: {metrics["MRR"]:.6f}')
    print(f'  Hits@1: {metrics["Hits@1"]:.6f}')
    print(f'  Hits@5: {metrics["Hits@5"]:.6f}')
    print(f'  Hits@10: {metrics["Hits@10"]:.6f}')
    print(f'  Accuracy: {metrics["Accuracy"]:.6f}')


def process_text_file(text_file, split=False):
    data_array = []
    with open(text_file, 'r') as data_file:
        for line_no, data_line in enumerate(data_file.readlines(), start=1):
            data_line = data_line.strip()
            if data_line == '':
                continue
            columns = data_line.split('\t')
            if len(columns) == 2:
                hop = None
                question_text = columns[0]
                answer_text = columns[1]
            elif len(columns) == 3:
                hop = columns[0].strip()
                question_text = columns[1]
                answer_text = columns[2]
            else:
                raise ValueError(f"{text_file}:{line_no} expected 2 or 3 tab-separated columns, got {len(columns)}")
            question = question_text.split('[', 1)
            if len(question) != 2:
                raise ValueError(f"{text_file}:{line_no} question is missing a [head_entity] marker")
            question_1 = question[0]
            question_2 = question[1].split(']', 1)
            if len(question_2) != 2:
                raise ValueError(f"{text_file}:{line_no} question is missing a closing ] for the head entity")
            head = question_2[0].strip()
            question_2 = question_2[1]
            question = question_1 + 'NE' + question_2
            ans = answer_text.split('|')
            data_point = [head, question.strip(), ans]
            if hop is not None:
                data_point.append(hop)
            data_array.append(data_point)
    if split is False:
        return data_array

    data = []
    for line in data_array:
        head = line[0]
        question = line[1]
        tails = line[2]
        hop = line[3] if len(line) > 3 else None
        for tail in tails:
            data_point = [head, question, tail]
            if hop is not None:
                data_point.append(hop)
            data.append(data_point)
    return data


def data_generator(data, word2ix, entity2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        encoded_question = encode_question(data_sample[1], word2ix)
        hop = data_sample[3] if len(data_sample) > 3 else None
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield (
            torch.tensor(head, dtype=torch.long),
            torch.tensor(encoded_question, dtype=torch.long),
            ans,
            torch.tensor(len(encoded_question), dtype=torch.long),
            data_sample[1],
            hop,
        )


def evaluate(data_path, device, model, word2idx, entity2idx):
    model.eval()
    data = process_text_file(data_path)
    data_gen = data_generator(data=data, word2ix=word2idx, entity2idx=entity2idx)

    answers = []
    metric_counts = init_metric_counts()
    hop_metric_counts = {}

    with torch.no_grad():
        for _ in tqdm(range(len(data))):
            d = next(data_gen)
            head = d[0].to(device)
            question = d[1].to(device)
            ans = d[2]
            ques_len = d[3].unsqueeze(0)
            q_text = d[4]
            hop = d[5]

            scores = model.get_score_ranked(head=head, sentence=question, sent_len=ques_len)[0]
            mask = torch.zeros(len(entity2idx)).to(device)
            mask[head] = 1
            new_scores = scores - (mask * 99999)
            pred_ans = torch.argmax(new_scores).item()

            hit_at_1 = in_top_k(new_scores, ans, 1)
            hit_at_5 = in_top_k(new_scores, ans, 5)
            hit_at_10 = in_top_k(new_scores, ans, 10)
            reciprocal_rank_value = reciprocal_rank(new_scores, ans)

            if type(ans) is int:
                ans = [ans]
            is_correct = 1 if pred_ans in ans else 0

            update_metric_counts(metric_counts, is_correct, hit_at_1, hit_at_5, hit_at_10, reciprocal_rank_value)
            if hop is not None:
                if hop not in hop_metric_counts:
                    hop_metric_counts[hop] = init_metric_counts()
                update_metric_counts(hop_metric_counts[hop], is_correct, hit_at_1, hit_at_5, hit_at_10, reciprocal_rank_value)

            answers.append(q_text + '\t' + str(pred_ans) + '\t' + str(is_correct))

    overall_metrics = finalize_metric_counts(metric_counts)
    hop_metrics = {hop: finalize_metric_counts(counts) for hop, counts in hop_metric_counts.items()}
    return answers, overall_metrics, hop_metrics


def get_chk_suffix():
    return '.chkpt'


def get_checkpoint_file_path(chkpt_path, model_name, num_hops, suffix, kg_type):
    return f"{chkpt_path}{model_name}_{num_hops}_{suffix}_{kg_type}"


def default_checkpoint_path(checkpoint_dir, model_name, num_hops, kg_type):
    checkpoint_base = get_checkpoint_file_path(str(checkpoint_dir) + os.sep, model_name, num_hops, '', kg_type)
    return Path(checkpoint_base + '_best_score_model' + get_chk_suffix())


def load_model_checkpoint(model, checkpoint_file):
    state = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
    model_state = model.state_dict()
    word_key = 'word_embeddings.weight'
    if word_key in state and state[word_key].shape != model_state[word_key].shape:
        old_weight = state[word_key]
        new_weight = model_state[word_key].clone()
        if old_weight.dim() == 2 and new_weight.dim() == 2 and old_weight.size(1) == new_weight.size(1) and old_weight.size(0) < new_weight.size(0):
            new_weight[:old_weight.size(0)] = old_weight
            state[word_key] = new_weight
            print(f'Expanded checkpoint {word_key} from {tuple(old_weight.shape)} to {tuple(new_weight.shape)}')
        else:
            raise ValueError(
                f"Checkpoint {word_key} has shape {tuple(old_weight.shape)}, "
                f"but current model expects {tuple(new_weight.shape)}. "
                f"Use the same vocabulary data file that was used during training."
            )
    model.load_state_dict(state)


def resolve_device(gpu, use_cuda):
    if use_cuda:
        if not torch.cuda.is_available():
            print('CUDA requested but unavailable; using CPU.')
            return torch.device('cpu')
        return torch.device(f'cuda:{gpu}')
    return torch.device('cpu')


def normalize_hops_for_data(hops):
    if hops in ['1', '2', '3', 'n']:
        return hops + 'hop'
    return hops


def default_data_path(qa_dataset, split, hops, kg_type, rephrased=False):
    data_hops = normalize_hops_for_data(hops)
    dataset_dir = qa_dataset
    if rephrased and split == 'test':
        dataset_dir = qa_dataset + '_rephrased'
    suffix = '_half' if split == 'train' and kg_type == 'half' else ''
    return REPO_ROOT / 'data' / 'QA_data' / dataset_dir / f'qa_{split}_{data_hops}{suffix}.txt'


def default_embedding_folder(model_name, kgembd_checkpoint_folder, kg_type):
    return REPO_ROOT / 'pretrained_models' / 'embeddings' / f'{model_name}_{kgembd_checkpoint_folder}_{kg_type}'


def load_bn_list(embedding_folder):
    bn_list = []
    for i in range(3):
        bn = np.load(embedding_folder / f'bn{i}.npy', allow_pickle=True)
        bn_list.append(bn.item())
    return bn_list


def build_model(args, device, embedding_folder, vocab_size, entity_count, embedding_matrix, bn_list):
    return RelationExtractor(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        num_entities=entity_count,
        relation_dim=args.relation_dim,
        pretrained_embeddings=embedding_matrix,
        freeze=args.freeze,
        device=device,
        entdrop=args.entdrop,
        reldrop=args.reldrop,
        scoredrop=args.scoredrop,
        l3_reg=args.l3_reg,
        model=args.model,
        ls=args.ls,
        loss_type=args.loss_type,
        w_matrix=str(embedding_folder / 'W.npy'),
        bn_list=bn_list,
    )


def write_answers(answers, output_path):
    with open(output_path, 'w') as f:
        for line in answers:
            f.write(line + '\n')
    print('Wrote predictions to', output_path)


def append_results(results_file, rows):
    write_header = not results_file.exists() or results_file.stat().st_size == 0
    fieldnames = ['KG-Model', 'KG-Type', 'hops', 'MRR', 'Hits@1', 'Hits@5', 'Hits@10', 'Accuracy']
    with open(results_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print('Appended metrics to', results_file)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a pretrained LSTM EmbedKGQA model.')
    parser.add_argument('--hops', type=str, default='1')
    parser.add_argument('--model', type=str, default='Rotat3')
    parser.add_argument('--kg_type', type=str, default='half')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--relation_dim', type=int, default=30)
    parser.add_argument('--entdrop', type=float, default=0.0)
    parser.add_argument('--reldrop', type=float, default=0.0)
    parser.add_argument('--scoredrop', type=float, default=0.0)
    parser.add_argument('--l3_reg', type=float, default=0.0)
    parser.add_argument('--ls', type=float, default=0.0)
    parser.add_argument('--freeze', type=str2bool, default=True)
    parser.add_argument('--qa-dataset', type=str, required=True, help='Dataset to use for evaluation, e.g. MetaQA or mquake.')
    parser.add_argument('--kgembd-checkpoint-folder', type=str, default=None, help='Embedding folder middle name. Defaults to --qa-dataset.')
    parser.add_argument('--loss_type', type=str, default='auto', choices=['auto', 'bce', 'kge'])
    parser.add_argument('--rephrased', action='store_true', help='Use rephrased test questions.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'dev', 'test'], help='QA split to evaluate.')
    parser.add_argument('--data-path', type=Path, default=None, help='Override the QA split file to evaluate.')
    parser.add_argument('--vocab-data-path', type=Path, default=None, help='QA file used to rebuild the training vocabulary. Defaults to qa_train_<hops>.txt.')
    parser.add_argument('--embedding-folder', type=Path, default=None, help='Override the pretrained KG embedding folder.')
    parser.add_argument('--checkpoint-file', type=Path, default=None, help='Override the QA model checkpoint file.')
    parser.add_argument('--checkpoint-dir', type=Path, default=None, help='Override the QA checkpoint directory.')
    parser.add_argument('--results-file', type=Path, default=Path('final_results.csv'), help='CSV file for metric output. Use --no-save-results to disable.')
    parser.add_argument('--no-save-results', action='store_true', help='Do not append metrics to a CSV file.')
    parser.add_argument('--answers-file', type=Path, default=None, help='Optional file for question, predicted entity id, and correctness rows.')
    return parser.parse_args()


def main():
    args = parse_args()
    qa_dataset = args.qa_dataset
    kgembd_checkpoint_folder = args.kgembd_checkpoint_folder or qa_dataset
    if args.loss_type == 'auto':
        args.loss_type = 'kge' if qa_dataset.lower() == 'mquake' else 'bce'

    embedding_folder = args.embedding_folder or default_embedding_folder(args.model, kgembd_checkpoint_folder, args.kg_type)
    vocab_data_path = args.vocab_data_path or default_data_path(qa_dataset, 'train', args.hops, args.kg_type)
    eval_data_path = args.data_path or default_data_path(qa_dataset, args.split, args.hops, args.kg_type, args.rephrased)
    checkpoint_dir = args.checkpoint_dir or (REPO_ROOT / 'checkpoints' / qa_dataset)
    checkpoint_file = args.checkpoint_file or default_checkpoint_path(checkpoint_dir, args.model, args.hops, args.kg_type)

    print('KG type is', args.kg_type)
    print('QA loss type is', args.loss_type)
    print('Embedding folder is', embedding_folder)
    print('Vocabulary data is', vocab_data_path)
    print('Evaluation data is', eval_data_path)
    print('Checkpoint file is', checkpoint_file)

    entities = np.load(embedding_folder / 'E.npy')
    relations = np.load(embedding_folder / 'R.npy')
    entity_embeddings, _ = preprocess_entities_relations(
        embedding_folder / 'entities.dict',
        embedding_folder / 'relations.dict',
        entities,
        relations,
    )
    entity2idx, _, embedding_matrix = prepare_embeddings(entity_embeddings)
    vocab_data = process_text_file(vocab_data_path, split=False)
    word2ix, _, _ = get_vocab(vocab_data)
    bn_list = load_bn_list(embedding_folder)

    device = resolve_device(args.gpu, args.use_cuda)
    model = build_model(args, device, embedding_folder, len(word2ix), len(entity2idx), embedding_matrix, bn_list)
    load_model_checkpoint(model, checkpoint_file)
    model.to(device)

    # Display model parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    answers, overall_metrics, hop_metrics = evaluate(
        data_path=eval_data_path,
        word2idx=word2ix,
        entity2idx=entity2idx,
        device=device,
        model=model,
    )

    print_test_metrics(f'[{args.split.capitalize()} overall]', overall_metrics)
    for hop in sorted(hop_metrics, key=hop_sort_key):
        print_test_metrics(f'[{args.split.capitalize()} {hop}-hop]', hop_metrics[hop])

    results = [{
        'KG-Model': args.model,
        'KG-Type': args.kg_type,
        'hops': args.hops,
        **overall_metrics,
    }]
    for hop in sorted(hop_metrics, key=hop_sort_key):
        results.append({
            'KG-Model': args.model,
            'KG-Type': args.kg_type,
            'hops': f'{hop}hop',
            **hop_metrics[hop],
        })

    if args.answers_file is not None:
        write_answers(answers, args.answers_file)
    if not args.no_save_results:
        append_results(args.results_file, results)


if __name__ == '__main__':
    main()
