# Inference on Test Data
We test the full model, regularized and trained for five epochs, on several out-of-distribution texts fethced from the test corpus. We let the model predict a highlight from the article, and wise versa. Also, we let it predict a *recovered* highlight from the predicted article, and compare it to the original highlight.

The articles has been chosen for being brief, but has otherwise not been cherry picked. See the [notebook](https://github.com/KristianWold/sum-and-elab/blob/main/notebooks/inference/test_data.ipynb)

## London Terror

**Real Article**: london (cnn)a 19-year-old man was charged wednesday with terror offenses after he was arrested as he returned to britain from turkey, london's metropolitan police said. yahya rashid, a uk national from northwest london, was detained at luton airport on tuesday after he arrived on a flight from istanbul, police said. he's been charged with engaging in conduct in preparation of acts of terrorism, and with engaging in conduct with the intention of assisting others to commit acts of terrorism. both charges relate to the period between november 1 and march 31. rashid is due to appear in westminster magistrates' court on wednesday, police said. cnn's lindsay isaac contributed to this report.

**Real Highlight**: london's metropolitan police say the man was arrested at luton airport after landing on a flight from istanbul. he's been charged with terror offenses allegedly committed since the start of november.

**Highlight Predicted from Article**: 

**Article Predicted from Highlight**: (cnn) -- a man who was arrested at luton airport on wednesday after landing on a flight from istanbul said he was arrested and questioned by an immigration officer. the man, who did not want to be named, was detained by immigration authorities. the man is charged with terrorism offenses, according to immigration and customs enforcement (fpf). he was booked on an unrelated charge. the man, originally from syria, was arrested wednesday at luton airport, where he boarded a flight from istanbul. he was charged with terror offenses, and the flight landed at luton international airport. he was arrested with suspected terrorism offenses, according to the airport. the latest charge is connected to the alleged plot in istanbul, the airport's website says.

**Recoverd Highlight**: a man on a flight from istanbul says he was arrested and questioned by immigration officers. he boarded the flight from istanbul to luton, authorities say. the man is charged with terrorism offenses, according to the airport's website.

### Comments

When predicting an article from the highlight, the model can represent the stated information correctly or incorrectly, in addition to inventing new information. 

Correct information:
 - Arrested at Luton airport
 - Came from Istanbul with plane

Incorrect information:
 - Prediction claimed suspect for terror, while article claims suspect for terror preparation.

Invented information:
 - Person from Syria
 - Arrest happened on Wednesday (possible mixup, the court was on wednesday)
 - Was questioned by immigration officer. 
 - Mention of unrelated charge.
 - Mention of a "plot" in Istanbul

Other Comments: 
 - Article is repetitive. Arrest in Luton is mentioned many times. 

## Ebola Infection

**Real Article**: (cnn)five americans who were monitored for three weeks at an omaha, nebraska, hospital after being exposed to ebola in west africa have been released, a nebraska medicine spokesman said in an email wednesday. one of the five had a heart-related issue on saturday and has been discharged but hasn't left the area, taylor wilson wrote. the others have already gone home. they were exposed to ebola in sierra leone in march, but none developed the deadly virus. they are clinicians for partners in health, a boston-based aid group. they all had contact with a colleague who was diagnosed with the disease and is being treated at the national institutes of health in bethesda, maryland. as of monday, that health care worker is in fair condition. the centers for disease control and prevention in atlanta has said the last of 17 patients who were being monitored are expected to be released by thursday. more than 10,000 people have died in a west african epidemic of ebola that dates to december 2013, according to the world health organization. almost all the deaths have been in guinea, liberia and sierra leone. ebola is spread by direct contact with the bodily fluids of an infected person.

**Real Highlight**: 17 americans were exposed to the ebola virus while in sierra leone in
march. another person was diagnosed with the disease and taken to hospital in
maryland. national institutes of health says the patient is in fair condition
after weeks of treatment.

**Highlight Predicted from Article**:

**Article Predicted from Highlight**: (cnn) -- four americans who were exposed to the deadly ebola virus in liberia, the health department said monday. one of the infected was in a tent at the liberian capital of san antonio, according to the centers for disease control and prevention. one of the men was treated at a medical center in bellevue, maryland, and is now in a stable condition, the health department said. the second man, who was treated at a medical center in maryland in late march, is in fair condition, the health department said. the other two is in stable condition. the other two are with suspected cases of ebola in the united states. the latest patient, who was treated at a hospital in monrovia, maryland, is in fair condition, the health department said. the two have been identified as 21-year-old sierra leonean thomas eric duncan. duncan is in a critical condition and is in a stable condition, the health department said. he is in a stable condition with three other patients who have died in the west african country, the health department said. the case has been referred to the national institutes of health.



**Recoverd Highlight**: a third patient is in stable condition, the health department says. five of the infected are in the liberian capital. one of the men was treated in a hospital in the united states. the outbreak has sickened more than 150 people in west africa.


## The Price is Right

**Real Article**: (cnn)for the first time in eight years, a tv legend returned to doing what he does best. contestants told to "come on down!" on the april 1 edition of "the price is right" encountered not host drew carey but another familiar face in charge of the proceedings. instead, there was bob barker, who hosted the tv game show for 35 years before stepping down in 2007. looking spry at 91, barker handled the first price-guessing game of the show, the classic "lucky seven," before turning hosting duties over to carey, who finished up. despite being away from the show for most of the past eight years, barker didn't seem to miss a beat.

**Real Highlight**: bob barker returned to host "the price is right" on wednesday. barker, 91, had retired as host in 2007.

**Highlight Predicted from Article**:

**Article Predicted from Highlight**: (cnn) -- he's a 92-year-old former member of the house of representatives. in the final episode of "the price is right" on the show wednesday night, the 91-year-old former member of the house of representatives was pictured in a photograph with fellow members of the house of representatives. he did not want to be identified. barker, an 83-year-old former member of the house of representatives, was seen with a photo of his grandson, who was present and his wife, the woman she was meeting. the couple left the house in 2007, but it did not stop there. barker, an older man of 83, had retired as host in 2007, according to his publicist. he later made a name for himself as a guest host on the house committee. he had previously served as the house representative on the house committee. barker had retired as host on the house committee and had been married in 2007. he has two daughters from his previous marriages with three daughters.


**Recoverd Highlight**: djerry barker leaves house of representatives in 2007. he had retired as host on house committee. he served as house representative on house committee. he has previously served as house representative on house committee.




