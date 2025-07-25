# Adaptive Patch-Size Dark Channel Prior Network (APSDCP-Net)

## Requirements

```
conda create -n apsdcp-net python=3.8
conda activate apsdcp-net
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Checkpoints

See [ckpts](https://github.com/mingyang-tu/apsdcp-net/tree/master/ckpts).

## Training

- Stage 1
    ```
    python src/train1.py -c configs/RESIDE-6K/train1.yml
    ```

- Stage 2
    ```
    python src/train2.py -c configs/RESIDE-6K/train2.yml
    ```

## Testing

```
python src/test.py -c configs/RESIDE-6K/test.yml
```

## Evaluation

```
python src/eval.py /path/to/dehazed/outputs/ /path/to/GT/
```
