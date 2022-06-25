from ai.transformer import data as D


def wrong_token_probability(model, dataset):
    wrong_tokens = 0
    for sample in dataset:
        # Prepare shapes in the dimensions of a one-item batch
        # [1, 14]
        src = sample["src"].unsqueeze(0)
        src_mask = D.make_src_mask(src)
        # [1, 13]
        target = sample["target"].unsqueeze(0)
        target_mask = D.make_target_mask(target)
        # [1, 13]
        target_y = sample["target_y"].unsqueeze(0)
        transformed = model(src, target, src_mask, target_mask)
        yhat = model.generator(transformed)

        err_ids = target_y != yhat.argmax(-1)
        wrong_tokens += err_ids.sum()

    all_tokens = len(dataset) * dataset.sequence_length
    mistake_probability = wrong_tokens / all_tokens

    return mistake_probability.item()
