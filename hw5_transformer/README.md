# Transformer language model homework

This homework has three parts.

1. In the first part of the homework (`01_multi_head_attention.ipynb`), you will implement the key concept of Transformer: multi-head attention.
1. In the second part, you need to use your multi-head attention implementation and `torch.nn` layers to implement Transformer Encoder. Your starter code with detailed task explanations is located in python file (not a notebook) `transformer_lm/modeling_transformer.py`. After implementing the model, you can test it in `02_transformer.ipynb`.
2. Implement training loop in `03_training.ipynb` and train 10 character-level language models.

## When do I need a GPU?

Only for the third part of the homework. Parts 1 and 2 do not involve training and can be done even a laptop CPU.

## Setting up the environment

All required libraries are listed in `requirements.txt`. You can install them using `pip install -r requirements.txt`.

Feel free to use Colab or Jupyter Lab for the first part of the homework, but we stongly recommend to use a code editor like VSCode or PyCharm for the second part, as it involves more interaction with `.py` files. Here is a good tutorial on how to [setup VSCode for Python](https://www.youtube.com/watch?v=Z3i04RoI9Fk). Both of them also support jupyter notebooks, you just need to specify which jupyter kernel you want to use (most probably its `nlp_class`). For VSCode you may want to additionally install a [markdown extention](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) to render files like this README.md.

**Monitor your model performance while it is training** in WadnB. This is the main purpose of this tool. If the model is not improving at all or is diverging, look up our "my model does not converge" checklist from Lecture 3. At the same time, if you get a very high test accuracy, your model might be cheating and your causal masking or data preprocessing is not implemented correctly. To help you understand what correct training loss plot and eval perplixity/accuracy should look like, we will post a couple of images in Slack. Your runs will most probaly look different, because of different hyperparemeters, but they should not be extremely different.

## Submitting this homework

> NOTE: Do not add `model.pt` and other large files to your submission.

1. Restart your `.ipynb` notebooks (part 1, part 2, and interact) and reexecute them top-to-bottom via "Restart and run all" button.
Not executed notebooks or the notebooks with the cells not executed in order will receive 0 points.
1. Add your report PDF to the homework contentes.
1. **If you are using GitHub**, add `github.com/guitaricet` and `github.com/NamrataRShivagunde` to your [repository collaborators](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-user-account/managing-access-to-your-personal-repositories/inviting-collaborators-to-a-personal-repository), so we could access your code. Push your last changes (inclusing the reexecuted notebooks) and subimt the link to your GitHub repository to the Blackboard.
2. **If you are not using GitHub**, delete `wandb` diretory it it was created. Zip this directory and submit the archive it to the Blackaord.
3. Submit a link to your best wandb project for this homewrok (name it "transformer_lm" or something like this) to the Blackboard.
