import os
import sys
import torch
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline


class Summarizer:
    def __init__(self, model_name="google/flan-t5-small"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_tokenized_data(self, data_dir, mode, prefix="summarize biomedical text in layman terms: "):

        def preprocess_function(examples):
            summaries = examples["lay_summary"]
            examples = examples["article"]
            # for i in range(len(examples)):
            #     examples[i] = examples[i].split('\n')[0]
            inputs = [prefix + doc for doc in examples]
            inputs = self.tokenizer(inputs, max_length=512, truncation=True)
            targets = self.tokenizer(text_target=summaries, max_length=512, truncation=True)
            inputs['labels'] = targets.input_ids
            return inputs

        data_path = os.path.join(data_dir, "eLife_" + mode + ".jsonl")
        dataset_1 = pd.read_json(data_path, lines=True)
        data_path = os.path.join(data_dir, "PLOS_" + mode + ".jsonl")
        dataset_2 = pd.read_json(data_path, lines=True)
        dataset = pd.concat([dataset_1, dataset_2], ignore_index=True)
        dataset = Dataset.from_pandas(dataset).shuffle()
        tokenized_dataset = dataset.map(preprocess_function, remove_columns=['lay_summary', 'article', 'headings', 'keywords', 'id'], batched=True)
        return tokenized_dataset


    def train(self, data_dir, save_dir):

        print("running_train")
        print(data_dir)
        print(save_dir)

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model_name)
        peft_config = LoraConfig(task_mode=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

        rouge = evaluate.load("rouge")
        bertscore = evaluate.load("bertscore")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            bert_results = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
            bert_precision = np.mean(bert_results["precision"])
            bert_recall = np.mean(bert_results["recall"])
            bert_f1 = np.mean(bert_results["f1"])
            bert_results = {
                'bert_precision': bert_precision,
                'bert_recall': bert_recall,
                'bert_f1': bert_f1
            }
            
            result = rouge_results | bert_results
            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
            result["gen_len"] = np.mean(prediction_lens)

            return {k: round(v, 4) for k, v in result.items()}

        tokenized_train_dataset = self.load_tokenized_data(data_dir, 'train')
        tokenized_val_dataset = self.load_tokenized_data(data_dir, 'val')
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model = get_peft_model(model, peft_config)

        training_args = Seq2SeqTrainingArguments(
            output_dir=save_dir,
            logging_dir=save_dir,
            evaluation_strategy="epoch",
            learning_rate=1e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=6,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False,
            report_to="none",
            save_strategy="epoch",
            load_best_model_at_end=True,
            generation_max_length=512
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        model.save_pretrained(save_dir)
        # tokenizer.save_pretrained("/kaggle/working/tokenizer_" + model_name)
        model.config.save_pretrained(save_dir)


    def predict(self, data_dir, model_dir, save_dir):
        
        print("running_test")
        print(data_dir)
        print(model_dir)
        print(save_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"    
        summarizer = pipeline("summarization", model=model_dir, tokenizer=self.model_name, truncation=True, device=device)

        def test(test_path, output_path):
            df = pd.read_json(test_path, lines=True)
            # df = pd.DataFrame(df.article)
            dataset = Dataset.from_pandas(df)
            inputs = dataset["article"]

            # BATCH_SIZE = 100
            with open(output_path, "w") as f:
                summaries = summarizer(inputs, max_length=512, min_length=256, do_sample=False)
                for summary in summaries:
                    f.write(summary['summary_text'] + "\n")
                # for input in inputs:
                #     summary = summarizer(input, max_length=512, min_length=256, do_sample=False)[0]['summary_text']
                # for i in range(0, len(inputs), BATCH_SIZE):
                #     last_ind = min(len(inputs), i + BATCH_SIZE)
                #     summaries = summarizer(inputs[i:last_ind], max_length=512, min_length=256, do_sample=False)
                #     for summary in summaries:
                #         f.write(summary['summary_text'] + "\n")

        data_path = os.path.join(data_dir, "eLife_test.jsonl")
        output_path = os.path.join(save_dir, "elife.txt")
        test(data_path, output_path)
        data_path = os.path.join(data_dir, "PLOS_test.jsonl")
        output_path = os.path.join(save_dir, "plos.txt")
        test(data_path, output_path)


if __name__=='__main__':
    mode = sys.argv[1]
    summ = Summarizer(model_name="google/flan-t5-small")
    
    if mode == 'train':
        data_dir = sys.argv[2]
        save_dir = sys.argv[3]
        summ.train(data_dir, save_dir)
    else:
        data_dir = sys.argv[2]
        model_dir = sys.argv[3]
        save_dir = sys.argv[4]
        summ.predict(data_dir, model_dir, save_dir)
