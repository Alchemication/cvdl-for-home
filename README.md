# cvdl-for-home
Computer Vision and Deep Learning for Anomaly Detection in Home Monitoring

### Instructions to convert chapters to a single pdf:

- Execute convert script (takes ~20 seconds)

```bash
git clone https://github.com/Alchemication/cvdl-for-home.git
cd ./cvdl-for-home/Scripts
./generate_pdf.sh
```

This will export a single *Thesis.ipynb* and *Thesis.pdf* files into a *Merged* folder


### First time setup

In order to successfully execute the shell script above, this one-time setup needs to be completed first:

- Create file ~/.jupyter/jupyter_nbconvert_config.py:

```python
c = get_config()
c.TemplateExporter.template_path = ['.', "~/.jupyter" ]
c.LatexExporter.template_path = ['.', "~/.jupyter"]
c.TemplateExporter.exclude_input_prompt = False # The default
c.PDFExporter.exclude_input_prompt = True
```

- Create template file ~/.jupyter/hidecode.tplx:

```
((*- extends 'article.tplx' -*))

((* block input_group *))
    ((*- if cell.metadata.get('nbconvert', {}).get('show_code', False) -*)) % hides code
            ((( super() )))
    ((*- endif -*))
((* endblock input_group *))

((* block commands *))
    \setcounter{secnumdepth}{0} % Turns off numbering for sections
    ((( super() )))
((* endblock commands *))
```
