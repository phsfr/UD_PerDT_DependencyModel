from hazm import *
import argparse


def produce_conllu_line(input_str):
    conllu_lines = []
    empty_tags = ['_'] * 8
    normalizer = Normalizer()
    input_str = normalizer.normalize(input_str)
    sents = sent_tokenize(input_str)
    for sent in sents:
        tokens = word_tokenize(sent)
        correct_toks = [tok.split('_') if '_' in tok else tok for tok in tokens]
        for idx, tok in enumerate(correct_toks):
            if isinstance(tok, list):
                for sub_tok in tok:
                    conllu_lines.append(str(idx + 1) + '\t' + sub_tok + '\t' + '\t'.join(empty_tags) + '\n')
            else:
                conllu_lines.append(str(idx + 1) + '\t' + tok + '\t' + '\t'.join(empty_tags) + '\n')
        conllu_lines.append('\n')
    return conllu_lines


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Input file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    args = parser.parse_args()

    input_file = args.input_file
    output_conllu_file = args.output_file
    f_w = open(output_conllu_file, 'w', encoding='utf-8')

    if input_file.endswith('.txt'):
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip('\n').strip()) == 0:
                    continue
                conllu_lines = produce_conllu_line(line)
                for conllu_line in conllu_lines:
                    f_w.write(conllu_line)
    f_w.close()
