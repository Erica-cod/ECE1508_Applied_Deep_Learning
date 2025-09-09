# 可以在这里写一下我们打算做什么

## 我了解到现在的热点：大模型生成式推荐

### idea1 推荐系统
可参展summary论文：Friend's recommendation on social media using different algorithms of machine learning  
https://www.sciencedirect.com/science/article/pii/S2666285X21000406

优点：公开数据集多，思路比较成熟。

缺点：比较同质化（做这个的人太多）

### idea2 我本科做过一段时间的：基于情感分析的隐喻识别
写的论文摘要：本研究探讨了结合情感分析和预训练语言模型（如RoBERTa和Transformer）进行隐喻识别的方法。隐喻作为自然语言中的一种复杂修辞手段，其识别对自然语言处理（NLP）提出了挑战。我们通过构建一个两阶段的模型，首先利用RoBERTa进行情感分析，随后在情感分析的基础上，通过Transformer模型识别文本中的隐喻。实验结果表明，RoBERTa+Transformer模型在准确率、召回率和F1值上均优于传统深度学习模型。尤其是在处理复杂隐喻和非显性隐喻特征的句子时，该模型表现出色。然而，模型在处理包含成语和生僻词汇的文本时存在局限性。本文研究表明，情感信息的辅助可以有效提升隐喻识别的准确性，为相关的自然语言处理任务提供了新的思路。

优点：有点创新性，算是科研创新点

缺点：这个项目有点烂尾，因为没有太合适的数据集，需要自己构建数据集进行训练

GPU要求：小批次的训练在4070laptop上面就可以跑

曾经参考的论文：BERT for Sentiment Analysis and Metaphor Recognition

Metaphor Identification in Discourse: A Corpus-Based Approach

### idea2的相关资源（这个我是问的GPT）：

“隐喻识别程序”（Metaphor Identification Procedure，简称 MIP），其核心思想是通过比较词汇在特定语境中的字面意义和上下文意义，来判断其是否具有隐喻性。

MIP 的基本步骤：
识别关键词：首先，识别出文本中可能具有多重意义的词汇。

比较意义：分析这些词汇在特定上下文中的意义，判断其是否偏离字面意义。

判断隐喻性：如果词汇在上下文中的意义明显不同于其字面意义，则认为该词汇具有隐喻性。
