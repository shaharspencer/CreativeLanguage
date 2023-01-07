
def difference_in_dobj_pobj():


    text_1 = "It's already been nine days. I'm amazed. It doesn't really feel like nine days at  all  - what it  really  feels like is nine days or nine seconds, all at the same time. My mind's been awhirl all that time, always rushing from one activity to the next. Our time is highly regulated; we'll have anywhere from 4-6 hours a day of lectures, and an equal amount of time spent in activities. We're on from 8:30 in the morning till 11 at night, no exceptions. Shad Valley is all about excellence, afterall. Some of the lectures are incredibly boring and excellent at inducing heavy amounts of sleep. I'll tell you guys all about it later, but right now I have to go to a Diaster Fair. What's a Diaster Fair, you ask? I have no idea. Check in later.  Cheers from Carleton, - Cary   Back Again   I suppose I should talk a little bit about what we actually do here at Shad Valley. Of course I mention the lectures and the activities, but I don't mention what those actually  are , what their point is. All these activities and workshops and lectures are supposed to be helping us with one of two things: our teambuilding skills, or the DE Project. Since most of you know what teambuilding skills are, I'll go into what the DE Project is.  Every year Shad Valley participants are assigned a theme. It's common to all campuses aross Canada; we all have the same amount of time to work on it, same amount of resources, etc. We have to design a device that deals with this theme. Not only do we have to design the device, but also a working prototype, a business plan, a corporate website, a marketing package, and a promotional package. It's a fair bit of work for three weeks. No one really knows what the 'DE' in 'the DE Project' stands for. Some speculate that it stands for 'disaster and emergency,' this year's theme. Others believe that it stands for 'design and engineering.' But no one knows for sure.   We only get about 2 hours of DE Project time a day. Considering the amount of work we have to do, that's not a lot of time. Not a lot at all. So we must be fairly efficient. That's where the teambuilding skills come into play. If you can't work together as a team you're going to fail. Plain and simple. Luckily, my house work together incredibly well. We may not have the coolest people, or the most technically knowledgeable, or the most physically skilled; but, we do have the house that works best together.  The most important part of our team dynamic is how we listen. Every meeting the house has, whether it be regarding the DE Project or not, we assign a different chairperson to chair the meeting. The chair's responsibilities are to basically let only one person talk at a time and keep the others quiet. This gives everyone in the group a leadership role at one time or another, and gives those who would not otherwise make themselves heard a voice.  It didn't used to be this way, however.  Prior to the rotating chair system, we relied on two leaders: myself, and my friend from Nanaimo, Stephen. Being a leader among leaders is quite the feeling, let me tell you. The system worked well enough: there was order and control, and everyone was heard. However, it didn't really reflect the values of Shad, and that wasn't a good thing.  Anyhow, it's about 8:21, and I have to get to a lecture for 8:30. More to come later.  Cheers from Stormont, - Cary"
    doc = nlp(text_1)

    doc_1 = list(doc.sents)[27]
    doc_2 = nlp(
        "Considering the amount of work we have to do, that's not a lot of time.")

    print(doc_1.text == doc_2.text)

    print(doc_1[0])
    print([child.dep_ for child in doc_1[0].children] == ["pobj"])
    print(doc_2[0])
    print([child.dep_ for child in doc_2[0].children] == ["dobj"])

if __name__ == '__main__':
    import spacy
    nlp = spacy.load('en_core_web_sm')
    difference_in_dobj_pobj()

