from datasets import load_metric
import pickle

eval_metric_accuracy = load_metric("accuracy")
eval_metric_f1 = load_metric("f1")
eval_metric_recall = load_metric("recall")
eval_metric_precision = load_metric("precision")

with open('dataset/predictions.pkl', 'rb') as f:
    predictions = pickle.load(f)

with open('dataset/labels.pkl', 'rb') as f:
    labels = pickle.load(f)

for i in range(len(predictions)):

    out_label = '0' if predictions[i][0] < 0.5767 else '1'

    eval_metric_accuracy.add(predictions=out_label, references=labels[i])
    eval_metric_f1.add(predictions=out_label, references=labels[i])
    eval_metric_recall.add(
        predictions=out_label, references=labels[i])
    eval_metric_precision.add(
        predictions=out_label, references=labels[i])
    # progress_bar_test.update(1)

print()
print(eval_metric_accuracy.compute())
print(eval_metric_precision.compute(average=None, zero_division=0))
print(eval_metric_recall.compute(average=None, zero_division=0))
print(eval_metric_f1.compute(average=None))
