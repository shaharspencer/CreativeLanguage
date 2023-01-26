import spacy
import pdb
pdb.set_trace()
lm = "en_core_web_lg"

# spacy.cli.download(lm)
# spacy.cli.download(lm)
import pathlib
def render(sent, output_path = "sent_1.svg"):
  from spacy import displacy

  nlp=spacy.load(lm)
  doc=nlp(sent)
  displacy.render(doc,jupyter=True,style="dep")
  svg = spacy.displacy.render(doc, style="dep")
  output_path = pathlib.Path(output_path)
  output_path.open('w', encoding="utf-8").write(svg)


render("Wake at 5:00 to be at work by 6:00 then get off at 3:30 and work on the house.")