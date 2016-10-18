# Face Recognition

```
sudo make bash
./model_1.py /root/openface/images/examples/adams.jpgAlign '/root/openface/images/examples/adams.jpg'
```

# Examples
From [OpenFace docs](https://github.com/cmusatyalab/openface/blob/master/docs/setup.md):

```
docker pull bamos/openface
docker run -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash
cd /root/openface
./demos/compare.py --verbose images/examples/{lennon*,clapton*}
```
