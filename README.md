# Chinese chess model train

* Learn how to use AI to train/play chinese chess, use CNN and Transformer to train.
* Use doubao AI, chatgpt AI and deepseek AI to help writting code and solve problem.

# AI model score/evaluation

* Doubao is good for simple requirements and simple code.
* Chatgpt is good at framework and design.
* Deepseek V4 is good at checking and analyzing code and local implementations.

# Training data

* Option 1：CGLemon/chinese-chess-PGN（GitHub，99k chess games，ICCS format）
https://github.com/CGLemon/chinese-chess-PGN
Download directly: click on green “Code”→Download ZIP
Format: .pgn / .iccs，parse by code directly

* Option 2：ModelScope (20000k chess games，SQLite）
https://www.modelscope.cn/datasets/nowcan/xiangqi_train_data
free to download, fit for bigger scope training

# Python verion

* Install any version of Python between 3.10.11～3.10.15

Download from：https://www.python.org/downloads/release/python-31011/
or use conda to install

# Dependent python libraries

``` bash
pip install -r requirements.txt
```
