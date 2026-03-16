"""
core/value_seeds.py

Seed sentences for the 15 Ethos value prototypes.

These sentences are embedded by BGE-large and averaged into a prototype
vector per value. The prototype is stored in Qdrant and used at query
time to score passage semantic similarity.

Design principles:
  1. Each seed shows the value in action, not as a dictionary definition.
     "He stood alone before the tribunal" beats "courage means being brave."

  2. Seeds span modern AND archaic register. Historical documents use
     "probity", "fortitude", "valour" — the prototype must bridge both.

  3. Seeds span first-person, third-person, and biographical narrative.
     The pipeline encounters all three in historical corpora.

  4. Seeds include values demonstrated under pressure — the resistance
     signal matters more than comfortable virtue performance.

  5. Seeds do NOT include failure examples. Failure detection uses
     the keyword layer and APY markers, not prototype similarity.

  6. 25-35 seeds per value. Enough for a stable centroid; more is
     diminishing returns after ~40.
"""

from __future__ import annotations
from typing import Dict, List

SEED_SENTENCES: Dict[str, List[str]] = {

    "integrity": [
        # Modern register — first-person
        "I told them the truth even though it cost me the promotion.",
        "I refused to sign the document because it contained falsehoods.",
        "I cannot pretend I did not know what was happening.",
        "I had to be honest with myself before I could be honest with anyone else.",
        "I gave my word and I kept it, regardless of the consequences.",
        "Even under pressure I would not say what I did not believe.",
        "I could not in good conscience remain silent about what I witnessed.",
        # Modern register — third-person / biographical
        "He told his superiors what they did not want to hear, knowing it would end his career.",
        "She refused to falsify the data, even when her funding depended on it.",
        "He acknowledged the error publicly rather than allowing the false record to stand.",
        "She kept her promise to the workers even after the political winds shifted.",
        # Under pressure / adversity
        "Despite every incentive to lie, he maintained the truth before the inquiry.",
        "She told the court exactly what she had seen, even though her testimony was unwelcome.",
        "He would not alter his account to protect those in power.",
        "Under threat of dismissal she refused to recant her findings.",
        # Archaic / historical register
        "He was a man of unimpeachable probity who would not bend his conscience to political convenience.",
        "Her rectitude in the face of inducement was remarked upon by all who knew her.",
        "He possessed that rare candour which made him incapable of dissimulation.",
        "She conducted herself with a veracity that no inducement could compromise.",
        "His incorruptible nature meant that no bribe, however lavish, could alter his testimony.",
        "He stood by his account with a forthrightness that shamed those who had expected otherwise.",
        "She spoke plainly when silence or flattery would have served her better.",
        "He refused to dissemble before the king, knowing the cost of plain speaking.",
        # Demonstrated in writing / private record
        "In his private journal he recorded what had truly occurred, setting down the facts without embellishment.",
        "Her letter to her sister confessed the full truth of the matter, holding nothing back.",
    ],

    "courage": [
        # Modern register — first-person
        "I was afraid, but I stood up and spoke anyway.",
        "I knew the risks and I went forward regardless.",
        "I could have stayed silent but it would have made me a coward.",
        "I was terrified but I held my ground.",
        "Despite the danger I pressed on with what I believed was right.",
        # Modern register — third-person / biographical
        "He walked into the crowd knowing he might not come out alive.",
        "She published the report despite the threats against her family.",
        "He stood alone before the tribunal and refused to retract.",
        "She chose the harder road and faced the consequences without flinching.",
        "He confronted the authorities knowing it would mean imprisonment.",
        # Under pressure / adversity
        "Even as the walls closed in around him he did not yield.",
        "She held the line despite every pressure to retreat.",
        "He refused to flee when all around him had already run.",
        "Under fire she continued to advance toward those who needed her.",
        # Archaic / historical register
        "He was of a valiant and dauntless character, unflinching before the most daunting adversary.",
        "Her gallantry in the field of battle earned the admiration of all who witnessed it.",
        "He pressed forward with an audacity that confounded those who had expected him to yield.",
        "She displayed a fortitude in suffering that those around her could only marvel at.",
        "He was resolute where others had broken, intrepid where others had turned back.",
        "With undaunted spirit he stood his ground against those who wished him gone.",
        "She marched on in spite of every danger, her courage undiminished by hardship.",
        "He knew the cost and paid it without complaint, pressing forward with bold determination.",
        # Biographical narrative
        "The record shows that he advanced toward the danger when retreat was still possible.",
        "Witnesses described her composure under fire as extraordinary — she had not flinched.",
        "Despite the impossibility of the odds he led the charge, refusing the safe option.",
    ],

    "compassion": [
        # Modern register — first-person
        "I could not turn away from their suffering even when it would have been easier to.",
        "Their pain moved me deeply and I could not pretend otherwise.",
        "I sat with her through the night because no one should face that alone.",
        "I felt their grief as if it were my own and acted accordingly.",
        "I gave what I had because the need in front of me was real.",
        # Modern register — third-person / biographical
        "She knelt beside the wounded soldier though the battle was still raging around her.",
        "He gave up his meal so that the child would not go hungry.",
        "She devoted years to those others had abandoned, never counting the cost to herself.",
        "He wept openly for the suffering of those he had never met.",
        "She attended to the sick at great personal risk, refusing to be driven away.",
        # Under pressure / adversity
        "Even when mercy was politically costly he extended it without hesitation.",
        "She offered clemency where the crowd demanded punishment.",
        "Despite having every reason to harden himself he remained open to the suffering of others.",
        # Archaic / historical register
        "He was moved by a deep benevolence toward those the world had forgotten.",
        "She showed a tender mercy to those who had wronged her, receiving them with forbearance.",
        "His charity toward the destitute was not the charity of the comfortable but of one who understood hardship.",
        "She was possessed of such kindheartedness that no appeal to her compassion was ever turned away.",
        "He was stricken to see such misery and could not remain unmoved.",
        "Her sympathies were broad and her tenderness toward the unfortunate was without limit.",
        "He moved among the suffering with a gentleness and solicitude that astonished those who accompanied him.",
        # Biographical narrative
        "Those who knew her recalled how she was never indifferent to the pain of others.",
        "Letters from the period record his deep concern for those displaced by the conflict.",
        "She is remembered not for her power but for the mercy she extended to those beneath her.",
    ],

    "commitment": [
        # Modern register — first-person
        "I gave my word and I will see it through no matter what it costs.",
        "I will not abandon what I started, not when so many are counting on me.",
        "I promised and a promise is not conditional on convenience.",
        "I devoted my life to this and I am not going to walk away now.",
        "I will not quit before it is finished.",
        # Modern register — third-person / biographical
        "She stayed at her post for thirty years, through every difficulty and disappointment.",
        "He returned to the cause again and again even after it had cost him everything.",
        "She kept her pledge to the community long after anyone would have blamed her for leaving.",
        "He gave himself entirely to the work, holding nothing back.",
        "She refused to abandon the project when it became difficult and the supporters had gone.",
        # Under pressure / adversity
        "Even after the first defeat he regrouped and pressed on, refusing to accept that it was over.",
        "She would not rest until what she had sworn to do was done.",
        "He carried the cause forward despite illness, opposition, and the loss of allies.",
        # Archaic / historical register
        "He had taken his oath with full knowledge of the cost and he would not renege upon it.",
        "She was bound by a vow she had made in full sincerity and would not be released from it by circumstance.",
        "His steadfast dedication to the cause outlasted the commitment of those who had begun alongside him.",
        "She had pledged herself to this course and no reversal of fortune would turn her from it.",
        "He persevered through seasons of defeat with an unwavering conviction that the end would justify the sacrifice.",
        "She had devoted herself wholly and would not now withhold what remained.",
        # Biographical narrative
        "His letters from the period reveal a man who regarded his commitment as absolute and irrevocable.",
        "The record of her life is one of sustained dedication to a single cause over many decades.",
        "Those who worked with him remarked on his refusal to be deflected from what he had undertaken.",
    ],

    "patience": [
        # Modern register — first-person
        "I waited even when every instinct told me to act.",
        "I held back and let things develop rather than forcing a conclusion.",
        "I knew the time was not right and I did not rush it.",
        "I endured the wait because the right outcome required it.",
        "I resisted the pressure to move before the moment was ready.",
        # Modern register — third-person / biographical
        "She waited years for the opportunity rather than seizing a lesser one.",
        "He bided his time through decades of setback, preparing for the moment.",
        "She held the course steady when others were demanding immediate action.",
        "He endured the delay with a composure that came from understanding the larger picture.",
        "She let the situation unfold without intervention, trusting the process she had set in motion.",
        # Under pressure / adversity
        "Even as those around him clamored for action he held steady, refusing to be rushed.",
        "She maintained her equanimity through months of provocation that would have broken others.",
        "He did not act rashly when patience would serve better, though every instinct cried urgency.",
        # Archaic / historical register
        "He was a man of remarkable forbearance, slow to anger and unwilling to act in haste.",
        "Her long-suffering in the face of repeated provocation was a constant source of wonder.",
        "He bore the delay with a measured composure that spoke of deep inner discipline.",
        "She possessed a serenity that allowed her to endure years of opposition without bitterness.",
        "His temperance in the face of urgent circumstance distinguished him from those who had acted rashly.",
        "She waited with an equanimity that others mistook for indifference but was in truth profound self-mastery.",
        # Biographical narrative
        "Those who studied his career noted how often he had waited when lesser men would have acted.",
        "She is remembered for the years of quiet preparation that preceded her most decisive actions.",
    ],

    "responsibility": [
        # Modern register — first-person
        "I was responsible for what happened and I am not going to pretend otherwise.",
        "It was my failure and I own it completely.",
        "I should have acted sooner and the consequences of my delay are on me.",
        "I cannot excuse what I failed to do. It was my duty and I did not fulfill it.",
        "I accept the blame. It falls to me and I will not shift it.",
        # Modern register — third-person / biographical
        "She accepted responsibility for the decision rather than deflecting to those beneath her.",
        "He stood before the affected families and acknowledged his role without equivocation.",
        "She bore the full weight of the outcome rather than distributing blame.",
        "He refused to shelter himself behind process when his judgment had been the issue.",
        "She was the kind of leader who absorbed the consequences of her choices rather than passing them down.",
        # Under pressure / adversity
        "Even knowing what it would cost him personally, he did not attempt to evade accountability.",
        "She accepted the censure of her superiors without complaint, knowing it was deserved.",
        "He would not allow others to take blame for what had been his decision.",
        # Archaic / historical register
        "He was answerable for what had occurred and he accepted that obligation without demur.",
        "She acknowledged her culpability in plain terms, refusing the evasions that were available to her.",
        "He regarded himself as the steward of those who trusted him and accepted the duties of that position.",
        "She bore her responsibility as an obligation that could not be delegated or diminished.",
        "He had been given the charge and he would answer for what became of it.",
        "She acknowledged her obligation to those who had depended on her judgment.",
        # Biographical narrative
        "His private letters reveal a man who held himself to strict account for every decision made under his authority.",
        "She is recorded as saying that those in power must bear the burden of their choices without excuse.",
    ],

    "fairness": [
        # Modern register — first-person
        "I treated them all by the same standard regardless of who they were.",
        "I would not give one party advantages I was unwilling to extend to the other.",
        "It was unfair and I said so even though it was not in my interest to.",
        "I refused to judge by anything other than the evidence in front of me.",
        "I gave them what they were owed, no more and no less.",
        # Modern register — third-person / biographical
        "She applied the same rules to the powerful and the powerless alike.",
        "He refused to let his personal feelings affect the judgment he was obliged to make.",
        "She insisted that all parties receive equal consideration regardless of their standing.",
        "He ruled on the matter as if neither side were known to him.",
        "She denied the special treatment that was requested and held to the common standard.",
        # Under pressure / adversity
        "Even when the decision cost him politically he ruled according to what the evidence demanded.",
        "She made the impartial judgment rather than the expedient one.",
        "Despite pressure from those in power he applied the same standard he would have applied to anyone.",
        # Archaic / historical register
        "He was a man of scrupulous equity who would not allow personal connection to influence his judgment.",
        "She was known for the even-handedness with which she administered the affairs under her charge.",
        "His impartiality in matters of dispute was such that both parties trusted his judgment.",
        "She showed no favoritism in her rulings, giving each party what the case truly merited.",
        "He measured every man by the same standard and was deaf to appeals based on position or wealth.",
        "Her judgments were proportionate and righteous, untainted by the partialities that clouded others.",
        # Biographical narrative
        "Those who came before her knew they would receive neither more nor less than what their case deserved.",
        "The record of his tenure shows a consistent and impartial application of principle.",
    ],

    "gratitude": [
        # Modern register — first-person
        "I would not be here without what others gave me and I have never forgotten it.",
        "I owe a debt I cannot fully repay to those who believed in me when I had given them little reason to.",
        "I am grateful in a way I find difficult to express for what was done for me.",
        "I have been given more than I deserved and I am aware of that every day.",
        "I could not have done this without them and I will not pretend otherwise.",
        # Modern register — third-person / biographical
        "She returned years later to thank those who had helped her when she had nothing.",
        "He spoke of his teachers with a reverence that never diminished over the decades.",
        "She never forgot the people who had stood with her before she had any standing.",
        "He acknowledged his debts publicly and completely, naming those whose help had made him.",
        # Archaic / historical register
        "He recorded in his letters how deeply indebted he felt to those who had made his education possible.",
        "She was possessed of a genuine gratitude toward fortune that prevented her from taking her advantages lightly.",
        "He counted his blessings with the earnestness of one who knew how easily they might not have been given.",
        "She regarded what she had received as gift rather than entitlement and conducted herself accordingly.",
        "He spoke of his benefactors with a warmth and acknowledgment of obligation that moved those who heard him.",
        "She was a woman who understood herself to be beholden to those who had come before and built the path she walked.",
        # Biographical narrative
        "Letters from this period reveal how consistently he returned to acknowledge those whose help had shaped him.",
        "She is remembered by those who knew her as genuinely and humbly grateful for every advantage she had received.",
        "He spoke in later years of the formative generosity extended to him and the lifelong obligation he felt.",
    ],

    "curiosity": [
        # Modern register — first-person
        "I had to know why it worked the way it did — I could not leave the question unanswered.",
        "I pushed further into the problem because the surface answer was not enough.",
        "I was compelled to investigate even when the investigation was inconvenient.",
        "I asked the question no one else seemed to want to ask and I kept asking it.",
        "I could not let the mystery rest — not understanding was more uncomfortable than the effort of finding out.",
        # Modern register — third-person / biographical
        "He turned every conversation into an inquiry, interrogating assumptions wherever he found them.",
        "She read everything she could find on the subject and then went looking for what had not yet been written.",
        "He was drawn to the unsolved question the way others were drawn to the settled answer.",
        "She built her life around the pursuit of understanding in a field most had dismissed.",
        "He examined the evidence with an almost obsessive attention to what it might reveal.",
        # Under pressure / adversity
        "Even after his conclusions were rejected he continued to investigate, convinced there was more to find.",
        "She pursued the inquiry despite being told the question was not worth asking.",
        "He could not be dissuaded from the line of investigation once it had taken hold of him.",
        # Archaic / historical register
        "He was of an inquisitive temper and would pursue a question to its uttermost conclusion.",
        "She possessed a searching mind that could not be satisfied with received opinion.",
        "He had a deep fondness for inquiry and would spare no effort in the investigation of a subject that had seized his interest.",
        "She was driven by an intellectual restlessness that no amount of learning could fully satisfy.",
        "His curiosity was of that relentless kind that would not permit him to leave a question half-examined.",
        "She had a probing intelligence that delighted in uncovering what lay beneath the surface of things.",
        # Biographical narrative
        "Those who knew him described an intellect that was never at rest, always probing, always questioning.",
        "Her correspondence reveals someone who regarded the pursuit of understanding as one of the highest activities.",
    ],

    "resilience": [
        # Modern register — first-person
        "I was knocked down and I got back up. That is all there is to say.",
        "I kept going even when I had every reason to stop.",
        "The setback nearly broke me but I would not let it be the end.",
        "I came back from that loss and built something stronger than what I had lost.",
        "I refused to let what happened define me or stop me.",
        # Modern register — third-person / biographical
        "She rebuilt from nothing three times and each time built something better.",
        "He emerged from prison without bitterness and resumed the work.",
        "She carried on through losses that would have ended most people.",
        "He refused to be broken by what had been done to him.",
        "She got back to work before anyone thought she was ready and outperformed everyone.",
        # Under pressure / adversity
        "Despite everything that had been taken from him he returned to the cause.",
        "She weathered opposition that lasted decades without being deflected.",
        "He pressed on through hardship that destroyed those around him.",
        "Even in the darkest period she did not give up on what she had set out to do.",
        # Archaic / historical register
        "He was of a tenacious and unyielding nature that no adversity could permanently subdue.",
        "She possessed an indomitable spirit that rose again and again from the ruins of her plans.",
        "He weathered every reversal with a fortitude that those around him could only admire.",
        "She had endured more than most men and had emerged from each trial not broken but tempered.",
        "His unconquered spirit was the marvel of those who had watched him suffer.",
        "She persevered through seasons of failure with a grit that defied ordinary expectation.",
        "He was unbowed by what had fallen upon him — each blow seemed only to reinforce his determination.",
        # Biographical narrative
        "The record of these years is one of repeated catastrophe followed by repeated recovery.",
        "Those who knew her spoke of a quality of soul that could not be permanently diminished.",
    ],

    "love": [
        # Modern register — first-person
        "I would have given everything for them and I came close to doing exactly that.",
        "They were the reason I kept going when I had no other reason left.",
        "I loved them in the way that does not diminish with time or difficulty.",
        "What I felt for them was not comfort — it was the kind of love that costs something.",
        "I held onto them through everything because they were everything.",
        # Modern register — third-person / biographical
        "He sacrificed his career to remain close to those he loved when distance would have been easier.",
        "She gave up what she had built so that those she loved would not have to suffer.",
        "He stayed when leaving would have been easier, because the bond was more important than his ease.",
        "She had loved deeply and the loss of that love shaped everything that came after.",
        "He put them first in a way that left no doubt about what mattered most to him.",
        # Under pressure / adversity
        "Even when the relationship had cost him much he would not abandon what he felt.",
        "She chose love over safety and the record shows she never regretted it.",
        "His devotion did not diminish when the person he loved became difficult to love.",
        # Archaic / historical register
        "He bore toward his wife a devotion that neither time nor difficulty could erode.",
        "She had a deep and ardent love for her children that animated every decision she made.",
        "His affection for those closest to him was not a sentiment but a governing principle.",
        "She cherished those in her care with a tenderness that was the constant note of her character.",
        "He held his family dear with a reverence that bordered on the sacred.",
        "She was bound to those she loved with a fidelity that outlasted every test.",
        # Biographical narrative
        "Letters from this period reveal a profound and sustaining love that shaped his entire public life.",
        "Those who knew her in private spoke of the depth of her attachment to those she cared for.",
    ],

    "growth": [
        # Modern register — first-person
        "I was wrong about that and when I understood why, I changed.",
        "That experience broke something in me that needed breaking.",
        "I had to unlearn what I had been taught before I could understand what was true.",
        "I became someone different because I chose to examine what I had always assumed.",
        "I looked at my failures honestly and let them teach me.",
        # Modern register — third-person / biographical
        "He came out of that period a fundamentally different thinker than he had been going in.",
        "She revised her views completely in the light of new evidence and was not ashamed to say so.",
        "The transformation in his thinking over those years was total and deliberate.",
        "She sought out those who would challenge her assumptions rather than confirm them.",
        "He credited his failures as the most important part of his education.",
        # Under pressure / adversity
        "Even the years of imprisonment became a period of profound intellectual and moral development.",
        "She emerged from the crisis not diminished but deepened.",
        "He used the defeat to examine himself in ways that success would never have required.",
        # Archaic / historical register
        "He undertook the cultivation of his understanding with the discipline of one who regards the mind as a field to be tended.",
        "She progressed through a series of revisions in her thinking that represented genuine moral development.",
        "He had matured considerably in judgment and no longer held the opinions of his earlier years.",
        "She had refined her understanding through experience and honest reflection over many years.",
        "His development as a thinker was the project of a lifetime pursued with uncommon discipline.",
        "She sought always to enlighten herself, regarding ignorance as an affliction to be remedied.",
        # Biographical narrative
        "Those who knew him early and late remarked on how profoundly he had changed — not in character but in depth.",
        "Her journals document a sustained and serious process of self-examination and revision.",
    ],

    "independence": [
        # Modern register — first-person
        "I made the decision myself and I will stand behind it myself.",
        "I was not going to let someone else determine what I believed or what I did.",
        "I chose my own course even when everyone around me thought it was wrong.",
        "I answered to my own conscience on this and to no one else.",
        "I built what I built on my own terms and I would not trade that for anything.",
        # Modern register — third-person / biographical
        "She refused to allow her views to be shaped by those who controlled her funding.",
        "He carved a path independent of the establishment that had trained him.",
        "She broke with her party when her conscience required it, regardless of the consequences.",
        "He would not be controlled by those who expected deference in exchange for their support.",
        "She operated outside the existing structures because she would not compromise what she was trying to do.",
        # Under pressure / adversity
        "Even when dependence would have made his life easier he refused it.",
        "She maintained her autonomy in the face of pressure that would have bent most people.",
        "He would not be told what to conclude and resisted every attempt to direct his thinking.",
        # Archaic / historical register
        "He was a man of sovereign mind who would not bend his convictions to please those in authority.",
        "She had a fierce love of liberty that made submission to any outside authority deeply uncongenial.",
        "He was self-reliant to a degree that some mistook for arrogance but was in fact the product of long self-discipline.",
        "She guarded her independence of judgment as the most precious of her possessions.",
        "He had carved his own path where others had followed the established route.",
        "She answered to no authority but her own understanding and would not pretend otherwise.",
        # Biographical narrative
        "Those who worked with her knew that she would not be directed — that her conclusions were her own.",
        "His career was marked by a refusal to subordinate his judgment to institutional convenience.",
    ],

    "loyalty": [
        # Modern register — first-person
        "I stood by them when standing by them cost me something.",
        "I did not leave when leaving would have been easier and safer.",
        "I stayed because I had given my word and because they were worth it.",
        "I will not abandon them now simply because the situation has become difficult.",
        "I held faith with them through everything that came and I do not regret it.",
        # Modern register — third-person / biographical
        "She stayed with the cause long after its prospects had dimmed and most had walked away.",
        "He did not desert his colleague when the charges were made, though desertion would have protected him.",
        "She remained true to her commitments even when betrayal had become the practical option.",
        "He kept faith with those who had trusted him when betraying them would have saved him considerable difficulty.",
        "She stood with those who needed her to stand with them, at real cost to herself.",
        # Under pressure / adversity
        "Even after everything they had put him through he would not turn against them.",
        "She held the line when all around her were breaking faith with what they had pledged.",
        "He refused to defect even when offered terms that would have made his life far easier.",
        # Archaic / historical register
        "He was a man of unswerving fidelity who would not desert those to whom he had given his allegiance.",
        "She maintained her loyalty with a steadfastness that was remarked upon by all who knew her circumstances.",
        "His fealty to those who had trusted him was such that no inducement could shake it.",
        "She did not betray those who had confided in her, though to do so would have been greatly to her advantage.",
        "He was steadfast in his support and no reversal of fortune could shake his devotion.",
        "She kept faith with those she had pledged herself to, in the face of every pressure to the contrary.",
        # Biographical narrative
        "The record shows that he stood by his allies through the worst of it and never wavered.",
        "Those who depended on her knew with certainty that she would not abandon them when it became costly.",
    ],

    "humility": [
        # Modern register — first-person
        "I was wrong and I have to admit that clearly and without excuse.",
        "I came to understand that I had been mistaken and I changed my position.",
        "I cannot claim credit for what others built before me and made possible.",
        "I did not know as much as I thought I did and I needed to hear that.",
        "I made a serious error of judgment and I take full responsibility for it.",
        # Modern register — third-person / biographical
        "She publicly revised her earlier position when the evidence showed it was wrong.",
        "He acknowledged his mistake before those he had wronged rather than allowing time to bury it.",
        "She was the kind of leader who gave credit away freely and absorbed blame readily.",
        "He conceded the point to a younger thinker and changed his published view accordingly.",
        "She listened to the criticism and acted on it, rather than defending what had proved indefensible.",
        # Under pressure / adversity
        "Even with his reputation at stake he admitted the error rather than defending the indefensible.",
        "She revised her lifelong position when new evidence made it untenable, at considerable cost to her standing.",
        "He acknowledged that those he had once dismissed had been right.",
        # Archaic / historical register
        "He was a man who owned his errors without qualification and did not seek to minimize them.",
        "She possessed a modesty of self-estimation that was neither performance nor weakness.",
        "He conceded openly that he had been in error and set about making the necessary correction.",
        "She was not above acknowledging that those of lower station had seen more clearly than she had.",
        "He received correction with a grace that spoke of a genuine humility rather than a performed one.",
        "She had no excessive opinion of her own judgment and was the first to admit when it had failed her.",
        # Biographical narrative
        "Those who knew him spoke of a man who could be changed by argument and who had changed many times.",
        "Her letters from this period contain a remarkably honest self-assessment of where she had gone wrong.",
        "He is remembered as a figure who grew throughout his life precisely because he never stopped questioning himself.",
    ],

}


def get_seeds() -> Dict[str, List[str]]:
    """Return the full seed dictionary."""
    return SEED_SENTENCES


def seed_count_summary() -> Dict[str, int]:
    """Return {value_name: seed_count} for all 15 values."""
    return {k: len(v) for k, v in SEED_SENTENCES.items()}
