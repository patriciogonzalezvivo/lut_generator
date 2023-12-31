
# Install

```
conda create -n lut_generator python=3.9
conda activate lut_generator
pip install -r requirements.txt
```

# Create LLUT (`lut.png` and `lut.cube`) from image with color checker (SpyerCheckr24)

```bash
python from_checker.py -i <image>
```

# Create LUT from image

```bash
```


# References

* https://github.com/colour-science/colour-checker-detection/blob/master/colour_checker_detection/examples/examples_detection.ipynb
* https://github.com/steveseguin/color-grading/blob/master/spyder_24_color_card.ipynb
* https://github.com/michelerenzullo/LUTify
