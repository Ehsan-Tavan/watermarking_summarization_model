OUTPUT_TEXT_COLUMN_NAMES = [
    "baseline_completion",
    "no_wm_output",
    "w_wm_output",
]


def compute_z_scores(example, watermark_detector=None, args=None):
    # this just iterates the z-score function over the columns we want to compute z-scores for
    for col_name in OUTPUT_TEXT_COLUMN_NAMES:
        if col_name in example:
            example = compute_z_score(
                example, text_column_name=col_name, watermark_detector=watermark_detector, args=args
            )
    return example


def compute_z_score(
        example,
        text_column_name=None,
        watermark_detector=None,
        args=None,
        window_size=None,
        window_stride=None,
):
    # for now, don't get the green token mask
    # if we're using normalizers
    return_green_token_mask = args.return_green_token_mask

    input_text = example[text_column_name][0]
    error = False
    if input_text == "":
        error = True
    else:
        try:
            score_dict = watermark_detector.detect(
                input_text,
                # window_size=window_size,
                # window_stride=window_stride,
                return_green_token_mask=return_green_token_mask,
                return_prediction=False,
                # this conversion to "decision" only desired in demo context
                # convert_to_float=True,  # this helps with integrity under NaNs
                # return_z_at_T=COMPUTE_SCORES_AT_T,
            )
        except Exception as e:
            print(e)
            error = True
    if error:
        problem_text = f"'{input_text[:40]} {'[...]' if len(input_text) > 40 else ''}'"
        if args.verbose:
            print(
                f"{(f'Windowed({window_size})' if window_size else '')} "
                f"Detection error on text: {problem_text}"
            )
        # "Error string too short to compute metrics"
        score_dict = watermark_detector.dummy_detect(
            return_prediction=False,
            return_green_token_mask=return_green_token_mask,
            return_z_at_T=args.compute_scores_at_t,
        )

    # current detect logic causes issues bc it only reports this sometimes
    score_dict.pop("confidence", None)

    # replace every key name in score dict with the text_column_name + key name
    # and then add them to the example dict
    score_dict = {
        text_column_name
        + (f"_win{window_size}-{window_stride}" if window_size else "")
        + "_"
        + k: v
        for k, v in score_dict.items()
    }
    example.update(score_dict)
    return example
