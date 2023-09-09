# Baichuan2-Explained
Baichuan2代码的逐行解析版本，适合小白




* [src/](./src)
  * [utils/](./src/utils)
    * [common.py](./src/utils/common.py)
      * init_adapter（）
      * load_pretrained()
      * prepare_args()
    * [peft_trainer.py  （定义LogCallback、PeftTrainer）](./src/utils/peft_trainer.py)
    * [data_collator.py（DataCollatorForChatGLM类）](./src/utils/data_collator.py)
    * [seq2seq.py  （ComputeMetrics、Seq2SeqTrainerForChatGLM)](./src/utils/seq2seq.py)
  * [train_sft.py（导入DataCollatorForChatGLM、Seq2SeqTrainerForChatGLM)](./src/train_sft.py)
* [web_demo.py/](./web_demo.py)
  * [ads_generation.md（分布式运行范例）](./examples/ads_generation.md)
* [README.md](./README.md)



# CSDN彩色博客版：
* [src/](./ChatGLM-Efficient-Tuning-Explained/src)
  * [utils/](./ChatGLM-Efficient-Tuning-Explained/src/utils)
    * [common.py](./ChatGLM-Efficient-Tuning-Explained/src/utils/common.py)
    * [peft_trainer.py](./ChatGLM-Efficient-Tuning-Explained/src/utils/peft_trainer.py)
  * [CSDN彩色源码解析train_sft.py](https://zengxiaojian.blog.csdn.net/article/details/131458667)
* [README.md](./ChatGLM-Efficient-Tuning-Explained/README.md)

ChatGLM Efficient Tuning源码解析train_sft.py   https://zengxiaojian.blog.csdn.net/article/details/131458667


## 引用 - 源项目

```bibtex
@Misc{chatglm-efficient-tuning,
  title = {ChatGLM Efficient Tuning},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/ChatGLM-Efficient-Tuning}},
  year = {2023}
}
```
