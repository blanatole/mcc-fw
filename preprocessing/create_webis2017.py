import os
import pandas as pd
import argparse
import json
from sklearn.model_selection import train_test_split

def load_data(path):
    """Loads instances and truth data and merges them."""
    instances_path = os.path.join(path, 'instances.jsonl')
    with open(instances_path, 'r') as f:
        instances = [json.loads(line) for line in f]
    df_instances = pd.DataFrame(instances)

    truth_path = os.path.join(path, 'truth.jsonl')
    with open(truth_path, 'r') as f:
        truth = [json.loads(line) for line in f]
    df_truth = pd.DataFrame(truth)

    return pd.merge(df_instances, df_truth, on='id')

def make_splits(df: pd.DataFrame, seed: int = 42):
    """Performs an 80/10/10 stratified split."""
    df_copy = df.copy()
    train_df, temp_df = train_test_split(
        df_copy, test_size=0.2, random_state=seed, stratify=df_copy['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_df['label']
    )

    train_df = train_df.assign(split='train')
    val_df = val_df.assign(split='val')
    test_df = test_df.assign(split='test')
    return pd.concat([train_df, val_df, test_df], ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='data/WebisClickbaitCorpus2017', help='Root directory of the unzipped Webis-Clickbait-17 dataset')
    parser.add_argument('--output', default='data/data_key_webis2017.csv', help='Output key CSV path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    args = parser.parse_args()

    val_path = os.path.join(args.dataset_root, 'clickbait17-validation-170630')
    test_path = os.path.join(args.dataset_root, 'clickbait17-test-170720')

    df_val = load_data(val_path)
    df_test = load_data(test_path)

    df_full = pd.concat([df_val, df_test], ignore_index=True)

    df_full['tweet_id'] = df_full['id']
    df_full['text'] = df_full['postText'].apply(lambda x: ' '.join(x))
    df_full['label'] = df_full['truthClass'].apply(lambda x: 1 if x == 'clickbait' else 0)

    def resolve_image_path(row):
        media_file = row['postMedia'][0] if row['postMedia'] else None
        if not media_file:
            return ''
        # media_file is like 'media/photo_xyz.jpg', we need to strip 'media/'
        image_filename = os.path.basename(media_file)
        if row['id'] in df_val['id'].values:
            return os.path.join(val_path, 'media', image_filename)
        else:
            return os.path.join(test_path, 'media', image_filename)

    df_full['image'] = df_full.apply(resolve_image_path, axis=1)

    # Perform the new split
    df_split = make_splits(df_full, seed=args.seed)

    output_df = df_split[['tweet_id', 'text', 'image', 'label', 'split']]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Saved re-splitted data to {args.output}")
    print("New split counts:")
    print(output_df['split'].value_counts())

if __name__ == '__main__':
    main()

