# cvdl-for-home
Computer Vision and Deep Learning for Anomaly Detection in Home Monitoring

### Instructions to convert chapters to a single pdf:

- Make sure you have the project cloned
```bash
git clone https://github.com/Alchemication/cvdl-for-home.git
cd ./cvdl-for-home/Scripts
```
- Update `BASE_PATH` variable in *generate_pdf.sh* file to point to local folder
- Update `\bibliography` path in *thesis-format.tplx* file to point to local folder
- Execute convert script (takes ~20 seconds)
```bash
./generate_pdf.sh
```

This will export a single *Thesis.ipynb* and *Thesis.pdf* files into a *Merged* folder

### First time setup

In order to successfully execute the shell script above, this one-time setup needs to be completed first:

- Create file ~/.jupyter/jupyter_nbconvert_config.py:

```python
c = get_config()
c.TemplateExporter.exclude_input_prompt = False # The default
c.PDFExporter.exclude_input_prompt = True
```

### Citations

All citations are in the *citations.bib* file in the root of the project. Reference them from Notebooks as:
```html
<cite data-cite="granger2013">(Granger, 2013)</cite>
```
, where `granger2013` is a key in the *.bib* file
