# watermarking 

## Introduction
Large language models (LLMs) have revolutionized natural language processing, enabling them to perform a wide range of tasks, from writing documents to generating executable code and answering complex questions with astonishing human-like proficiency.

Considering these concerns, we focus our efforts in this repository on analyzing the paper [<b>A Watermark for Large Language Models</b>](https://arxiv.org/abs/2301.10226) and studying watermarking effect on the summarization task by using a language model. Through the implementation of this innovative algorithm, our objective is to effectively identify machine-generated text and prevent the potentially harmful effects of language models.


The Watermark algorithm used to generate token s<sup>t</sup> begins by calculating a hash of the preceding token, s<sup>t-1</sup>, and employs this hash as a seed for a random number generator. This seed is then utilized to generate two distinct sets: the "green list" (G) and the "red list" (R). During the generation of token 's<sup>t</sup>', the model is restricted to employing only those tokens present in the green list. Remarkably, because the red list is chosen at random, a natural writer is expected to violate the red list rule for approximately half of their tokens. In contrast, the watermarked model consistently avoids producing any violations.

## T5 Analysis
In this section, we analyze the impact of watermarking on the T5 summarization model. This model is available at [huggingface](https://huggingface.co/sysresearch101/t5-large-finetuned-xsum-cnn) repository and fine-tuned on [CNN Daily Mail](https://huggingface.co/datasets/cnn_dailymail), and [XSUM](https://huggingface.co/datasets/xsum) datasets.


To analyse watermarking on this model we randomly select the 1000 sample from test set of CNN Daily Mail dataset. 

The Watermark algorithm has two main parameters to create a red and green list, gamma and alpha. Gamma is the ratio of tokens to put in the green list when splitting the vocabulary and delta is the amount of bias (absolute) to add to the logits of the green list tokens at every step.



Table 1 presents the evaluation results of the "watermarks" algorithm when applied to a T5 fine-tuned summarization model. For each combination of "Gamma γ" and "Delta δ", the table reports the algorithm's performance under two distinct generation methods: "Greedy Search" and "Beam Search" decoding.

<table style='text-align:center;'>
  <tr>
    <td rowspan='2'> <b>Gamma γ</b> </td>
    <td rowspan='2'> <b>Delta δ</b> </td>
    <td colspan="2"><b>Rouge-1</b></td>
    <td colspan="2"><b>Rouge-2</b></td>
    <td colspan="2"><b>Rouge-L</b></td>
    <td colspan="2"><b>z-score</b></td>
  </tr>
  <tr>
    <td colspan="1"><b>Greedy</b></td>
    <td colspan="1"><b>Beam</b></td>
    <td colspan="1"><b>Greedy</b></td>
    <td colspan="1"><b>Beam</b></td>
    <td colspan="1"><b>Greedy</b></td>
    <td colspan="1"><b>Beam</b></td>
    <td colspan="1"><b>Greedy</b></td>
    <td colspan="1"><b>Beam</b></td>
  </tr>
  <tr>
  <td rowspan='4'>0.25</td>
    <td colspan="1"> 0(NW) </td>
    <td colspan="1"> 19.63 </td>
    <td colspan="1"> 20.82 </td>
    <td colspan="1"> 4.88 </td>
    <td colspan="1"> 5.42 </td>
    <td colspan="1"> 13.83 </td>
    <td colspan="1"> 14.4 </td>
    <td colspan="1"> - 0.11 </td>
    <td colspan="1"> - 0.32 </td>
  </tr>
  <tr>
    <td>2</td>
    <td colspan="1"> 19.34 </td>
    <td colspan="1"> 20.36 </td>
    <td colspan="1"> 3.92 </td>
    <td colspan="1"> 4.33 </td>
    <td colspan="1"> 13.17 </td>
    <td colspan="1"> 13.61 </td>
    <td colspan="1"> 3.37 </td>
    <td colspan="1"> 4.1 </td>
  </tr>
  <tr>
    <td colspan="1"> 5 </td>
    <td colspan="1"> 16.18 </td>
    <td colspan="1"> 14.62 </td>
    <td colspan="1"> 1.84 </td>
    <td colspan="1"> 1.59 </td>
    <td colspan="1"> 10.8 </td>
    <td colspan="1"> 8.98 </td>
    <td colspan="1"> 8.21 </td>
    <td colspan="1"> 8.51 </td>
  </tr>
  <tr>
    <td colspan="1"> 10 </td>
    <td colspan="1">  13.81 </td>
    <td colspan="1"> 10.94 </td>
    <td colspan="1"> 0.95 </td>
    <td colspan="1"> 0.85 </td>
    <td colspan="1"> 9.17 </td>
    <td colspan="1"> 6.58 </td>
    <td colspan="1"> 11.55 </td>
    <td colspan="1"> 7.96 </td>
  </tr>
<tr>
  <td rowspan='4'>0.5</td>
    <td colspan="1"> 0(NW) </td>
    <td colspan="1"> 19.63 </td>
    <td colspan="1"> 20.82 </td>
    <td colspan="1"> 4.88 </td>
    <td colspan="1"> 5.42 </td>
    <td colspan="1"> 13.83 </td>
    <td colspan="1"> 14.4 </td>
    <td colspan="1"> - 0.30 </td>
    <td colspan="1"> - 0.15 </td>
  </tr>
  <tr>
    <td>2</td>
    <td colspan="1"> 19.12 </td>
    <td colspan="1"> 21.6 </td>
    <td colspan="1"> 3.97 </td>
    <td colspan="1"> 4.59 </td>
    <td colspan="1"> 13.09 </td>
    <td colspan="1"> 13.83 </td>
    <td colspan="1"> 2.47 </td>
    <td colspan="1"> 3.42 </td>
  </tr>
  <tr>
    <td colspan="1"> 5 </td>
    <td colspan="1"> 17.5 </td>
    <td colspan="1"> 17.24 </td>
    <td colspan="1"> 2.61 </td>
    <td colspan="1"> 2.57 </td>
    <td colspan="1"> 11.59 </td>
    <td colspan="1"> 10.83 </td>
    <td colspan="1"> 4.6 </td>
    <td colspan="1"> 5.37 </td>
  </tr>
  <tr>
    <td colspan="1"> 10 </td>
    <td colspan="1"> 15.09 </td>
    <td colspan="1"> 13.38 </td>
    <td colspan="1"> 1.75 </td>
    <td colspan="1"> 1.7 </td>
    <td colspan="1"> 10.26 </td>
    <td colspan="1"> 8.27 </td>
    <td colspan="1"> 6.2 </td>
    <td colspan="1"> 5.06 </td>
  </tr>
  <tr>
  <td rowspan="4"> 0.75 </td>
    <td colspan="1"> 0(NW) </td>
    <td colspan="1"> 19.63 </td>
    <td colspan="1"> 20.82 </td>
    <td colspan="1"> 4.88 </td>
    <td colspan="1"> 5.42 </td>
    <td colspan="1"> 13.83 </td>
    <td colspan="1"> 14.4 </td>
    <td colspan="1"> 0.12 </td>
    <td colspan="1"> 0.11 </td>
  </tr>
  <tr>
    <td>2</td>
    <td colspan="1"> 19.15 </td>
    <td colspan="1">  20.79 </td>
    <td colspan="1"> 4.46 </td>
    <td colspan="1"> 5.15 </td>
    <td colspan="1"> 13.31 </td>
    <td colspan="1"> 14.41 </td>
    <td colspan="1"> 1.8 </td>
    <td colspan="1"> 2.36 </td>
  </tr>
  <tr>
    <td colspan="1"> 5 </td>
    <td colspan="1"> 18.46 </td>
    <td colspan="1"> 20.62 </td>
    <td colspan="1"> 3.69 </td>
    <td colspan="1"> 4.16 </td>
    <td colspan="1"> 12.77 </td>
    <td colspan="1"> 13.51 </td>
    <td colspan="1"> 2.61 </td>
    <td colspan="1"> 3.16 </td>
  </tr>
  <tr>
    <td colspan="1"> 10 </td>
    <td colspan="1"> 17.74 </td>
    <td colspan="1"> 19.82 </td>
    <td colspan="1"> 3.21 </td>
    <td colspan="1"> 3.77 </td>
    <td colspan="1"> 12.25 </td>
    <td colspan="1"> 12.87 </td>
    <td colspan="1"> 2.97 </td>
    <td colspan="1"> 3.34 </td>
  </tr>

</table>

<b style='text-align:center;'>Table 1: Summarization results for various γ and δ values.</b>

As shown in table 1, the model can achieve a very strong watermark (large z-score) by choosing a small green list size γ and a large green list bias δ. For instance, in our analysis, by selecting γ as 0.25 and δ as 10, we observe z-scores of 11.55 for greedy search and 7.96 for beam search. 

However, it is crucial to note that enhancing the watermark's strength can lead to potential distortion in the generated text.  For instance, within this configuration, the Rouge-1 scores for the greedy search method are 13.87 for γ set to 0.25, while increasing γ to 0.5 under the same δ value of 10,  the Rouge-1 score rises to 15.09. Furthermore, choosing γ to 0.75 results in a Rouge-1 score of 17.74 for the greedy search. In summary, if we aim for a very strong watermark, the performance of the model in the summarization task decreases.

Figure 1 represents the tradeoff between the z-score and rouge-1, considering various green list size ratio (γ). When examining the case of a small green list bias (δ=2), the alterations in the z-score and rouge-1 metrics exhibit a minimal changes to adjustments in the γ value. This is because the small delta cannot make a noticeable alteration in the softmax output value. Conversely, when a greater green list bias is used (δ=10), changing in γ lead to substantial alterations in both z-score and rouge-1 metrics.


| δ=2                            | δ=5                            | δ=10                            |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |
| ![ehsa](assets/plots/tradeoff_zscore_rouge_1_delta_2.png) | ![ehsa](assets/plots/tradeoff_zscore_rouge_1_delta_5.png) | ![ehsa](assets/plots/tradeoff_zscore_rouge_1_delta_10.png)|

<b style='text-align:center;'>Figure 1: Tradeoff between z-score and Rouge-1 with different δ.</b>

Figure 2 illustrates the relationship between the z-score and rouge-1 metrics, considering various levels of bias (δ) in the green token list.   When γ is set to 0.25, we observe the most substantial shifts in both z-score and rouge-1 metrics in response to alterations in δ. This is attributed to the limited size of the green token list, where even slight changes in δ can significantly impact model performance, leading to a decline in rouge-1 scores. Conversely, for γ set to 0.75, alterations in δ result in the smallest changes in z-score and the rouge-1, particularly when compared to the scenario where γ is 0.25. This outcome is due to the substantial size of the green token list, allowing the model to achieve satisfactory rouge-1 performance. Nevertheless, the larger scale of the green token list also corresponds to a relatively lower z-score value (weak watermark) compared to the situation when γ is 0.25.

| γ=0.25                            | γ=0.5                            | γ=0.75                            |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |
| ![ehsa](assets/plots/tradeoff_zscore_rouge_1_gamma_0.25.png) | ![ehsa](assets/plots/tradeoff_zscore_rouge_1_gamma_0.5.png) | ![ehsa](assets/plots/tradeoff_zscore_rouge_1_gamma_0.75.png)|

<b style='text-align:center;'>Figure 2: Tradeoff between z-score and Rouge-1 with different γ.</b>


<table style='text-align:center;'>
  <tr>
    <td rowspan='2'> <b>Gamma γ</b> </td>
    <td rowspan='2'> <b>Delta δ</b> </td>
    <td colspan="2"><b>Coherence</b></td>
    <td colspan="2"><b>Mauve</b></td>
    <td colspan="2"><b>z-score</b></td>
  </tr>
  <tr>
    <td colspan="1"><b>Greedy</b></td>
    <td colspan="1"><b>Beam</b></td>
    <td colspan="1"><b>Greedy</b></td>
    <td colspan="1"><b>Beam</b></td>
    <td colspan="1"><b>Greedy</b></td>
    <td colspan="1"><b>Beam</b></td>
  </tr>
  <tr>
  <td rowspan='4'>0.25</td>
    <td colspan="1"> 0(NW) </td>
    <td colspan="1"> 62.98 </td>
    <td colspan="1"> 63.04 </td>
    <td colspan="1"> 95.29 </td>
    <td colspan="1"> 96.71 </td>
    <td colspan="1"> -0.11 </td>
    <td colspan="1"> -0.32 </td>
  </tr>
  <tr>
    <td colspan="1"> 2 </td>
    <td colspan="1"> 61.25 </td>
    <td colspan="1"> 61.38 </td>
    <td colspan="1"> 95.68 </td>
    <td colspan="1"> 91.82 </td>
    <td colspan="1"> 3.37 </td>
    <td colspan="1"> 4.1 </td>
  </tr>
  <tr>
    <td colspan="1"> 5 </td>
    <td colspan="1"> 56 </td>
    <td colspan="1"> 57.63 </td>
    <td colspan="1"> 92.42 </td>
    <td colspan="1"> 67.97 </td>
    <td colspan="1"> 8.21 </td>
    <td colspan="1"> 8.51 </td>
  </tr>
  <tr>
    <td colspan="1"> 10 </td>
    <td colspan="1"> 52.11 </td>
    <td colspan="1"> 55.51 </td>
    <td colspan="1"> 87.54 </td>
    <td colspan="1"> 30.44 </td>
    <td colspan="1"> 11.55 </td>
    <td colspan="1"> 7.96 </td>
  </tr>
  <tr>
  <td rowspan='4'>0.5</td>
    <td colspan="1"> 0(NW) </td>
    <td colspan="1"> 62.98 </td>
    <td colspan="1"> 63.04 </td>
    <td colspan="1"> 95.29 </td>
    <td colspan="1"> 96.71 </td>
    <td colspan="1"> -0.30 </td>
    <td colspan="1"> -0.15 </td>
  </tr>
  <tr>
    <td colspan="1"> 2 </td>
    <td colspan="1"> 62.12 </td>
    <td colspan="1"> 62.97 </td>
    <td colspan="1"> 95.5 </td>
    <td colspan="1"> 94.73 </td>
    <td colspan="1"> 2.47 </td>
    <td colspan="1"> 3.42 </td>
  </tr>
  <tr>
    <td colspan="1"> 5 </td>
    <td colspan="1"> 60.05 </td>
    <td colspan="1"> 60.96 </td>
    <td colspan="1"> 94.28 </td>
    <td colspan="1"> 67.7 </td>
    <td colspan="1"> 4.6 </td>
    <td colspan="1"> 5.37 </td>
  </tr>
  <tr>
    <td colspan="1"> 10 </td>
    <td colspan="1"> 57.82 </td>
    <td colspan="1"> 59.91 </td>
    <td colspan="1"> 94.7 </td>
    <td colspan="1"> 32.34 </td>
    <td colspan="1"> 6.2 </td>
    <td colspan="1"> 5.06 </td>
  </tr>
  <tr>
  <td rowspan='4'>0.75</td>
    <td colspan="1"> 0(NW) </td>
    <td colspan="1"> 62.98 </td>
    <td colspan="1"> 63.04 </td>
    <td colspan="1"> 95.29 </td>
    <td colspan="1"> 96.71 </td>
    <td colspan="1"> 0.12 </td>
    <td colspan="1"> 0.11 </td>
  </tr>
  <tr>
    <td colspan="1"> 2 </td>
    <td colspan="1"> 62.73 </td>
    <td colspan="1"> 63.49 </td>
    <td colspan="1"> 96 </td>
    <td colspan="1"> 95.8 </td>
    <td colspan="1"> 1.8 </td>
    <td colspan="1"> 2.36 </td>
  </tr>
  <tr>
    <td colspan="1"> 5 </td>
    <td colspan="1"> 61.93 </td>
    <td colspan="1"> 62.48 </td>
    <td colspan="1"> 95.1 </td>
    <td colspan="1"> 95.91 </td>
    <td colspan="1"> 2.61 </td>
    <td colspan="1"> 3.16 </td>
  </tr>
  <tr>
    <td colspan="1"> 10 </td>
    <td colspan="1"> 61.01 </td>
    <td colspan="1"> 61.97 </td>
    <td colspan="1"> 91.48 </td>
    <td colspan="1"> 92.71 </td>
    <td colspan="1"> 2.97 </td>
    <td colspan="1"> 3.34 </td>
  </tr>
</table>

<b style='text-align:center;'>Table 2: Coherence and Mauve analysis on summarization task for various γ and δ values.
Models</b>




