function addQuestions(form){
  addQuestion1(form);
  addQuestion2(form);
  addQuestion3(form);

}

function addQuestion1(form){

        item = "1. Does the sentence feel creative?";
        var choices = [...Array(7).keys()];
        form.addMultipleChoiceItem().setTitle(item).setChoiceValues(choices);

        item = "1.a. If so: Characterize the creative linguistic element(s) that invoked a creative feeling. Only use quotations from the original sentence. You may pick more than one word. If you pick non-adjacent words, place a comma [,] between them."
        form.addParagraphTextItem().setTitle(item);


        item = "1.b. Please characterize the creativity type of the utterance in one word. (e.g. beautiful, aesthetic, humorous, idiomatic, word-play, cheeky, ironic etc.) ";
        form.addParagraphTextItem().setTitle(item);

        item = "1.c	Rephrase the clause to be as concise as possible.";
        form.addParagraphTextItem().setTitle(item);

}

function addQuestion2(form){
  item = "2. Does the sentence fit the nominal definition of the linguistic dimension? (yes or no)";
  form.addCheckboxItem().setTitle(item);
  item = "2.a	If not: elaborate.";
  form.addParagraphTextItem().setTitle(item);
}

function addQuestion3(form){
  item = "3. Did you notice any technical problems?"
  form.addCheckboxItem().setTitle(item);
  form.addParagraphTextItem().setTitle("if so elaborate");
}

function openCsv(){
   var sheet = SpreadsheetApp.openByUrl("https://docs.google.com/spreadsheets/d/1beKNFXWIuOQArpehQfOcxBYQB_bObfhppRQf6b1kI6c/edit");
 var data = sheet.getDataRange().getValues();
 var words  = [];
 var sents = [];
 data.forEach(function (row) {
   words.push(row[0]);
   sents.push(row[3]);
 });
  var lst = [words, sents]
  return lst;
}

function createForm() {

   // create & name Form
   var item = "Creative sents form";
   var form = FormApp.create(item)
       .setTitle(item);

  var lst = openCsv();
  for (var i = 1; i < 2; i++){
        // item.setTitle("general").
      const special_word = lst[0][i];
      special_word.getTextStyle().setBold(true);
      form.addSectionHeaderItem().setTitle("sentence # " + i + ",<b> " + special_word + "</b>: " + lst[1][i]);
      addQuestions(form);

  }

  form.setAllowResponseEdits(true);
  form.setCollectEmail(true);
  form.setConfirmationMessage("Thanks! have a great day :)");
  form.setProgressBar(true);
  form.setPublishingSummary(true);
  form.setShowLinkToRespondAgain(true);

}
