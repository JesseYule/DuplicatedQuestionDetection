<script type="text/javascript"
   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# DuplicateTextDetection
以Quora重复问题检测为例，利用Transformer作为特征提取器，构建ESIM模型进行分析。

###模型流程

1. 利用torchtext进行数据的预处理：

* 定义不同数据的处理操作；

* 加载数据；

* 创建词汇表，通过词汇表把词语和预训练的词向量连接起来；

* 将处理后的数据batch化；

2. 正式建模：

* 建立embedding层，对输入进行embedding；

* 利用Transformer对数据再次进行embedding，自注意力机制的利用为词向量引入上下文信息；

* 对向量进行局部推断，计算两个句子之间词与词的相似度：

  





第一次从零开始写一个项目，才发现虽然以前拿别人的代码改一下可以很快地实现各种模型，但未必真的熟悉了这个模型或者使用的框架。现在有各种各样的深度学习框架，我们可以很方便地构建模型，但是在具体到每一步时，是否知道应该做什么，是否知道用哪个框架的哪个函数实现，这些都是我开始写这个项目之后才开始思考的问题。我不是说只有不借助任何框架写一个模型才算真的深入理解这个模型，只是我们是不是真的有自己的一套建模的流程，这才是需要慢慢学习改善的。

以下我也列举一下我在建模时想到的一些小问题：

1. 这个模型需要对数据做padding吗，在embedding之前还是之后做padding？
2. 有必要处理句子中的标点符号吗，如何处理？
3. 反向传播对padding有影响吗，假如我们通过LSTM提取句子的上下文信息，之后得到的词向量也包括padding的部分，那么这部分对模型的学习有什么影响？
4. 句子长度差异过大时，对于较短的句子，padding对模型有影响吗？

