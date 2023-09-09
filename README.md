# Baichuan2-Explained
Baichuan2代码的逐行解析版本，适合小白




* [fine-tune/](./fine-tune)
  * [fine-tune.py/](./fine-tune/fine-tune.py)
    * class ModelArguments : const model_name_or_path
    * class DataArguments : const data_path
    * class TrainingArguments ： const cache_dir、const optim、const model_max_length、 const use_lora、
    * class SupervisedDataset
  * func train 
* [web_demo.py/](./web_demo.py)
  * [ads_generation.md（分布式运行范例）](./examples/ads_generation.md)
* [README.md](./README.md)



# CSDN彩色博客版：
* [src/](./ChatGLM-Efficient-Tuning-Explained/src)
  * [utils/](./ChatGLM-Efficient-Tuning-Explained/src/utils)
    * [common.py](./ChatGLM-Efficient-Tuning-Explained/src/utils/common.py)
    * [peft_trainer.py](./ChatGLM-Efficient-Tuning-Explained/src/utils/peft_trainer.py)
  * [CSDN彩色源码解析web_demo.py](https://blog.csdn.net/sinat_37574187/article/details/132779405?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22132779405%22%2C%22source%22%3A%22sinat_37574187%22%7D)
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
