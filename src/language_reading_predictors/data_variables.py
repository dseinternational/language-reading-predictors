# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later


class Variables:
    """
    Data variable names
    """

    SUBJECT_ID = "subject_id"
    """
    Unique subject identifier
    """

    TIME = "time"
    """
    Time point of measurement (1, 2, 3, 4)
    """

    GROUP = "group"
    """
    Intervention group (1: Initial intervention, from t1; 2: Wait for intervention, from t2)  
    """

    AREA = "area"
    """
    Study geographical areas (1: North, 2: South)
    """

    GENDER = "gender"
    """
    Gender of the subject (1: Boy, 2: Girl)
    """

    AGE = "age"
    """
    Age in months
    """

    BLOCKS = "blocks"
    """
    Block Design subtest from the Wechsler Preschool and Primary Scale of 
    Intelligence – Third Edition (WPPSI-III) (Wechsler, 2002). Only
    administered at time point 1.
    """

    APTGRAM = "aptgram"
    """
    Expressive grammar raw score. Assessed using the Action Picture Test 
    (Renfrew, 1997).
    """

    APTINFO = "aptinfo"
    """
    Expressive information raw score. Assessed using the Action Picture Test 
    (Renfrew, 1997).
    """

    B1EXTAU = "b1extau"
    """
    Block 1 Expressive Vocabulary Taught. Tests were created to measure expressive 
    and receptive knowledge of words explicitly taught in each phase of the 
    intervention. Six words of each type (nouns, adverbs, adjectives, prepositions) 
    were tested. In the expressive test, children were shown pictures that they were 
    asked to: name (nouns); say what the person was doing (verbs; e.g. 'what is the 
    man doing?' 'Stretching'); name after a prompt related to a comparison picture 
    (adjectives; e.g. 'this boy is clean, this boy is...?' 'Dirty'); or answer a 
    specific question designed to elicit a preposition (e.g. 'where is the book?' 
    'On the table').
    """

    B1EXTO = "b1exto"
    """
    Block 1 Expressive Vocabulary Total. Tests were created to measure expressive 
    and receptive knowledge of words explicitly taught in each phase of the 
    intervention. Six words of each type (nouns, adverbs, adjectives, prepositions) 
    were tested. In the expressive test, children were shown pictures that they were 
    asked to: name (nouns); say what the person was doing (verbs; e.g. 'what is the 
    man doing?' 'Stretching'); name after a prompt related to a comparison picture 
    (adjectives; e.g. 'this boy is clean, this boy is...?' 'Dirty'); or answer a 
    specific question designed to elicit a preposition (e.g. 'where is the book?' 
    'On the table').
    """

    B1EXNT = "b1exnt"
    """
    Block 1 Expressive Vocabulary Not Taught. Tests were created to measure expressive 
    and receptive knowledge of words explicitly taught in each phase of the 
    intervention. Six words of each type (nouns, adverbs, adjectives, prepositions) 
    were tested. In the expressive test, children were shown pictures that they were 
    asked to: name (nouns); say what the person was doing (verbs; e.g. 'what is the 
    man doing?' 'Stretching'); name after a prompt related to a comparison picture 
    (adjectives; e.g. 'this boy is clean, this boy is...?' 'Dirty'); or answer a 
    specific question designed to elicit a preposition (e.g. 'where is the book?' 
    'On the table').
    """

    B1RETAU = "b1retau"
    """
    Block 1 Receptive Vocabulary Taught. Tests were created to measure expressive 
    and receptive knowledge of words explicitly taught in each phase of the 
    intervention. In the receptive test, children were asked to select the picture
    (from a choice of 4) which represented the target word.
    """

    B1RETO = "b1reto"
    """
    Block 1 Receptive Vocabulary Total. Tests were created to measure expressive 
    and receptive knowledge of words explicitly taught in each phase of the 
    intervention. In the receptive test, children were asked to select the picture
    (from a choice of 4) which represented the target word.
    """

    B1RENT = "b1rent"
    """
    Block 1 Receptive Vocabulary Not Taught. Tests were created to measure expressive 
    and receptive knowledge of words explicitly taught in each phase of the 
    intervention. In the receptive test, children were asked to select the picture
    (from a choice of 4) which represented the target word.
    """

    B2EXTAU = "b2extau"
    """
    Block 2 Expressive Vocabulary Taught. Tests were created to measure expressive 
    and receptive knowledge of words explicitly taught in each phase of the 
    intervention. Six words of each type (nouns, adverbs, adjectives, prepositions) 
    were tested. In the expressive test, children were shown pictures that they were 
    asked to: name (nouns); say what the person was doing (verbs; e.g. 'what is the 
    man doing?' 'Stretching'); name after a prompt related to a comparison picture 
    (adjectives; e.g. 'this boy is clean, this boy is...?' 'Dirty'); or answer a 
    specific question designed to elicit a preposition (e.g. 'where is the book?' 
    'On the table').
    """

    B2EXTO = "b2exto"
    """
    Block 2 Expressive Vocabulary Total. Tests were created to measure expressive 
    and receptive knowledge of words explicitly taught in each phase of the 
    intervention. Six words of each type (nouns, adverbs, adjectives, prepositions) 
    were tested. In the expressive test, children were shown pictures that they were 
    asked to: name (nouns); say what the person was doing (verbs; e.g. 'what is the 
    man doing?' 'Stretching'); name after a prompt related to a comparison picture 
    (adjectives; e.g. 'this boy is clean, this boy is...?' 'Dirty'); or answer a 
    specific question designed to elicit a preposition (e.g. 'where is the book?' 
    'On the table').
    """

    B2EXNT = "b2exnt"
    """
    Block 2 Expressive Vocabulary Not Taught. Tests were created to measure expressive 
    and receptive knowledge of words explicitly taught in each phase of the 
    intervention. Six words of each type (nouns, adverbs, adjectives, prepositions) 
    were tested. In the expressive test, children were shown pictures that they were 
    asked to: name (nouns); say what the person was doing (verbs; e.g. 'what is the 
    man doing?' 'Stretching'); name after a prompt related to a comparison picture 
    (adjectives; e.g. 'this boy is clean, this boy is...?' 'Dirty'); or answer a 
    specific question designed to elicit a preposition (e.g. 'where is the book?' 
    'On the table').
    """

    B2RETAU = "b2retau"
    """
    Block 2 Receptive Vocabulary Taught. Tests were created to measure expressive 
    and receptive knowledge of words explicitly taught in each phase of the 
    intervention. In the receptive test, children were asked to select the picture
    (from a choice of 4) which represented the target word.
    """

    B2RETO = "b2reto"
    """
    Block 2 Receptive Vocabulary Total. Tests were created to measure expressive 
    and receptive knowledge of words explicitly taught in each phase of the 
    intervention. In the receptive test, children were asked to select the picture
    (from a choice of 4) which represented the target word.
    """

    B2RENT = "b2rent"
    """
    Block 2 Receptive Vocabulary Not Taught. Tests were created to measure expressive 
    and receptive knowledge of words explicitly taught in each phase of the 
    intervention. In the receptive test, children were asked to select the picture
    (from a choice of 4) which represented the target word.
    """

    CELF = "celf"
    """
    Basic concept knowledge score from the Clinical Evaluation of Language 
    Fundamentals Preschool 2nd Edition (Wiig, Secord, & Semel, 2006).
    Assessed knowledge of 18 basic linguistic concepts.
    """

    EOWPVT = "eowpvt"
    """
    Expressive One-Word Picture Vocabulary Test (Brownell, 2000) raw score.
    """

    ERBNW = "erbnw"
    """
    Early Repetition Battery nonword raw score
    """

    ERBTO = "erbto"
    """
    Early Repetition Battery total raw score
    """

    ERBWORD = "erbword"
    """
    Early Repetition Battery word raw score
    """

    LSAMMAX = "lsammax"
    """
    Language sample maximum length utterance
    """

    LSAMMLU = "lsammlu"
    """
    Language sample mean length utterance
    """

    LSAMINT = "lsamint"
    """
    Language sample percent intelligible words
    """

    LSAMUN = "lsamun"
    """
    Language sample total unique words
    """

    LSAMTO = "lsamto"
    """
    Language sample total words
    """

    NONWORD = "nonword"
    """
    Nonword reading. Children were asked to read the names of six cartoon 
    monsters: 'et', 'om', 'ip', 'neg', 'sab' and 'hic'. This test was devised 
    because all available nonword reading tests were too difficult. Two 
    practice items were given before test items.
    """

    BLENDING = "blending"
    """
    Phoneme blending. The child was asked to select which of three pictures
    represented a word spoken by the experimenter in 'robot' talk. On each trial 
    a target picture (e.g. bed) was presented along with pictures representing 
    an initial phoneme distracter (e.g. bud) and a rhyming distracter 
    (e.g. head). All targets and distracters had a consonant-vowel-consonant 
    structure. Two practice items were followed by 10 test items.
    """

    ROWPVT = "rowpvt"
    """
    Receptive One-Word Picture Vocabulary Test (Brownell, 2000) raw score.
    """

    SPPHON = "spphon"
    """
    Phonetic spelling score.
    """

    SPRAW = "spraw"
    """
    Spelling raw score. Ten words were presented as pictures to be named and 
    spelled. If no letters were correctly represented in the first two items 
    the test was discontinued.
    """

    TROG = "trog"
    """
    Receptive grammar items correct score from the Test for Reception of 
    Grammar 2 (TROG-2; Bishop, 2003). Eight grammatical constructs were tested 
    in blocks of four items; each correct item was awarded a score of 1.
    """

    YARCEWR = "yarcewr"
    """ 
    Early Word Recognition (EWR) test from the York Assessment of Reading (YARC) 
    Early Reading battery (Hulme et al., 2009). Children reading over 25 words 
    were given an additional set of words from the Test of Single-Word Reading, 
    from the YARC.
    """

    YARCLET = "yarclet"
    """
    Extended test of alphabetic knowledge from the YARC (Hulme et al., 2009) asks 
    the child to provide the sound for 32 individual letters and digraphs.
    """

    YARCSI = "yarcsi"
    """
    Comprehension skills inventory from the York Assessment of Reading (YARC) 
    Early Reading battery (Hulme et al., 2009).
    """

    DEAPPIN = "deappin"
    """
    DEAP Picture Naming Initial
    """

    DEAPPVO = "deappvo"
    """
    DEAP Picture Naming Vowel
    """

    DEAPPFI = "deappfi"
    """
    DEAP Picture Naming Final
    """

    DEAPPAV = "deappav"
    """
    DEAP Picture Naming Average
    """

    DEAPP_C = "deapp_c"
    """
    DEAP Picture Composite Score: DEAPPIN + DEAPPVO + DEAPPFI
    """

    EWRSWR = "ewrswr"
    """
    Reading Composite Score EWR and SWR
    """

    TASCORE = "tascore"
    """
    TA Effectiveness (1=Excellent 3=Poor)
    """

    TACHANG = "tachang"
    """
    Number of TA Changes Between Time 1 and 2
    """

    BEHAV = "behav"
    """
    Behaviour (1: very good; 5: very challenging). Assessed at t1–t3 by ratings 
    of videorecordings of assessment sessions. Using a timesampling technique, 
    behaviour was rated over 10s periods on a 5-point scale (1 = very good; 
    5 = very challenging) every 5 min through 60 min of film; scores were 
    averaged to create a single score for each child at each time point.
    """

    ATTEND = "attend"
    """
    Attendance Up To Week
    """

    SDQ = "sdq"
    """
    Strengths and Difficulties Questionnaire
    """

    AGESPEAK = "agespeak"
    """ 
    Age when child started speaking
    """

    VISION = "vision"
    """ 
    Vision Normal or Impaired
    """

    HEARING = "hearing"
    """
    Hearing Normal or Impaired
    """

    EARINF = "earinf"
    """
    Repeated Bouts of Ear Infections
    """

    HEARING_C = "hearing_c"
    """
    Hearing Composite: HEARING || EARINF (0: normal hearing and no repeated ear 
    infections; 1: either impaired hearing or repeated ear infections)
    """

    NUMCHIL = "numchil"
    """
    Number of Children in the Family
    """

    MUMOCC = "mumocc"
    """
    Mum current occupation
    """

    DADOCC = "dadocc"
    """
    Dad current occupation
    """

    AGEBOOKS = "agebooks"
    """ 
    Age of child when first exposed to picture books
    """

    MUMEDUPOST16 = "mumedupost16"
    """
    Mother's years in full time education post-16.
    """

    DADEDUPOST16 = "dadedupost16"
    """
    Father's years in full time education post-16.
    """

    BEDTIMEREAD = "bedtimeread"
    """
    Frequency of reading at bedtime (0 - 6 times per week, or 7 or more).
    """

    OTHERTIMEREAD = "othertimeread"
    """
    Frequency of reading at other times (0 - 6 times per week, or 7 or more).
    """

    SPPHON_NONE = "spphon_none"
    """
    Never scored more than 0 on SPPHON at any time point (0: scored > 1 at at least one time point; 1: only 
    ever scored 0 or NaN at all time points).
    """

    NONWORD_NONE = "nonword_none"
    """
    Never scored more than 0 on NONWORD at any time point (0: scored > 1 at at least one time point; 1: only 
    ever scored 0 or NaN at all time points).
    """

    _NEXT_SUFFIX = "_next"

    EWRSWR_NEXT = EWRSWR + _NEXT_SUFFIX
    """
    EWRSWR score at the next time point (NaN for t4).
    """

    ERBNW_NEXT = ERBNW + _NEXT_SUFFIX
    """
    ERBNW score at the next time point (NaN for t4).    
    """

    ERBTO_NEXT = ERBTO + _NEXT_SUFFIX
    """
    ERBTO score at the next time point (NaN for t4).    
    """

    YARCEWR_NEXT = YARCEWR + _NEXT_SUFFIX
    """
    YARCEWR score at the next time point (NaN for t4).    
    """

    ERBWORD_NEXT = ERBWORD + _NEXT_SUFFIX

    ROWPVT_NEXT = ROWPVT + _NEXT_SUFFIX

    EOWPVT_NEXT = EOWPVT + _NEXT_SUFFIX

    YARCLET_NEXT = YARCLET + _NEXT_SUFFIX

    YARCSI_NEXT = YARCSI + _NEXT_SUFFIX

    SPPHON_NEXT = SPPHON + _NEXT_SUFFIX

    SPRAW_NEXT = SPRAW + _NEXT_SUFFIX

    BLENDING_NEXT = BLENDING + _NEXT_SUFFIX

    NONWORD_NEXT = NONWORD + _NEXT_SUFFIX

    TROG_NEXT = TROG + _NEXT_SUFFIX

    CELF_NEXT = CELF + _NEXT_SUFFIX

    APTGRAM_NEXT = APTGRAM + _NEXT_SUFFIX

    APTINFO_NEXT = APTINFO + _NEXT_SUFFIX

    B1EXNT_NEXT = B1EXNT + _NEXT_SUFFIX

    B1EXTO_NEXT = B1EXTO + _NEXT_SUFFIX

    B1EXTAU_NEXT = B1EXTAU + _NEXT_SUFFIX

    B1RENT_NEXT = B1RENT + _NEXT_SUFFIX

    B1RETO_NEXT = B1RETO + _NEXT_SUFFIX

    B1RETAU_NEXT = B1RETAU + _NEXT_SUFFIX

    B2EXNT_NEXT = B2EXNT + _NEXT_SUFFIX

    B2EXTO_NEXT = B2EXTO + _NEXT_SUFFIX

    B2EXTAU_NEXT = B2EXTAU + _NEXT_SUFFIX

    B2RENT_NEXT = B2RENT + _NEXT_SUFFIX

    B2RETO_NEXT = B2RETO + _NEXT_SUFFIX

    B2RETAU_NEXT = B2RETAU + _NEXT_SUFFIX

    DEAPPIN_NEXT = DEAPPIN + _NEXT_SUFFIX

    DEAPPVO_NEXT = DEAPPVO + _NEXT_SUFFIX

    DEAPPFI_NEXT = DEAPPFI + _NEXT_SUFFIX

    DEAPPAV_NEXT = DEAPPAV + _NEXT_SUFFIX

    LSAMMAX_NEXT = LSAMMAX + _NEXT_SUFFIX

    LSAMMLU_NEXT = LSAMMLU + _NEXT_SUFFIX

    LSAMINT_NEXT = LSAMINT + _NEXT_SUFFIX

    LSAMUN_NEXT = LSAMUN + _NEXT_SUFFIX

    LSAMTO_NEXT = LSAMTO + _NEXT_SUFFIX

    _GAIN_SUFFIX = "_gain"

    EWRSWR_GAIN = EWRSWR + _GAIN_SUFFIX
    """
    EWRSWR gain score from current to next time point.
    """

    ERBNW_GAIN = ERBNW + _GAIN_SUFFIX
    """
    ERBNW gain score from current to next time point.
    """

    ERBTO_GAIN = ERBTO + _GAIN_SUFFIX

    YARCEWR_GAIN = YARCEWR + _GAIN_SUFFIX

    ERBWORD_GAIN = ERBWORD + _GAIN_SUFFIX

    ROWPVT_GAIN = ROWPVT + _GAIN_SUFFIX

    EOWPVT_GAIN = EOWPVT + _GAIN_SUFFIX

    YARCLET_GAIN = YARCLET + _GAIN_SUFFIX

    YARCSI_GAIN = YARCSI + _GAIN_SUFFIX

    SPPHON_GAIN = SPPHON + _GAIN_SUFFIX

    SPRAW_GAIN = SPRAW + _GAIN_SUFFIX

    BLENDING_GAIN = BLENDING + _GAIN_SUFFIX

    NONWORD_GAIN = NONWORD + _GAIN_SUFFIX

    TROG_GAIN = TROG + _GAIN_SUFFIX

    CELF_GAIN = CELF + _GAIN_SUFFIX

    APTGRAM_GAIN = APTGRAM + _GAIN_SUFFIX

    APTINFO_GAIN = APTINFO + _GAIN_SUFFIX

    B1EXNT_GAIN = B1EXNT + _GAIN_SUFFIX

    B1EXTO_GAIN = B1EXTO + _GAIN_SUFFIX

    B1EXTAU_GAIN = B1EXTAU + _GAIN_SUFFIX

    B1RENT_GAIN = B1RENT + _GAIN_SUFFIX

    B1RETO_GAIN = B1RETO + _GAIN_SUFFIX

    B1RETAU_GAIN = B1RETAU + _GAIN_SUFFIX

    B2EXNT_GAIN = B2EXNT + _GAIN_SUFFIX

    B2EXTO_GAIN = B2EXTO + _GAIN_SUFFIX

    B2EXTAU_GAIN = B2EXTAU + _GAIN_SUFFIX

    B2RENT_GAIN = B2RENT + _GAIN_SUFFIX

    B2RETO_GAIN = B2RETO + _GAIN_SUFFIX

    B2RETAU_GAIN = B2RETAU + _GAIN_SUFFIX

    DEAPPIN_GAIN = DEAPPIN + _GAIN_SUFFIX

    DEAPPVO_GAIN = DEAPPVO + _GAIN_SUFFIX

    DEAPPFI_GAIN = DEAPPFI + _GAIN_SUFFIX

    DEAPPAV_GAIN = DEAPPAV + _GAIN_SUFFIX

    LSAMMAX_GAIN = LSAMMAX + _GAIN_SUFFIX

    LSAMMLU_GAIN = LSAMMLU + _GAIN_SUFFIX

    LSAMINT_GAIN = LSAMINT + _GAIN_SUFFIX

    LSAMUN_GAIN = LSAMUN + _GAIN_SUFFIX

    LSAMTO_GAIN = LSAMTO + _GAIN_SUFFIX

    ATTEND_CUMUL = "attend_cumul"
    """
    Cumulative attendance up to current time point.
    """

    ALL = [
        SUBJECT_ID,
        TIME,
        GROUP,
        AREA,
        GENDER,
        AGE,
        BLOCKS,
        APTGRAM,
        APTINFO,
        B1EXTAU,
        B1EXTO,
        B1EXNT,
        B1RETAU,
        B1RETO,
        B1RENT,
        B2EXTAU,
        B2EXTO,
        B2EXNT,
        B2RETAU,
        B2RETO,
        B2RENT,
        CELF,
        EOWPVT,
        ERBNW,
        ERBTO,
        ERBWORD,
        LSAMMAX,
        LSAMMLU,
        LSAMINT,
        LSAMUN,
        LSAMTO,
        NONWORD,
        BLENDING,
        ROWPVT,
        SPPHON,
        SPRAW,
        TROG,
        YARCEWR,
        YARCLET,
        YARCSI,
        DEAPPIN,
        DEAPPVO,
        DEAPPFI,
        DEAPPAV,
        DEAPP_C,
        EWRSWR,
        TASCORE,
        TACHANG,
        BEHAV,
        ATTEND,
        SDQ,
        AGESPEAK,
        VISION,
        HEARING,
        EARINF,
        HEARING_C,
        NUMCHIL,
        MUMOCC,
        DADOCC,
        AGEBOOKS,
        MUMEDUPOST16,
        DADEDUPOST16,
        BEDTIMEREAD,
        OTHERTIMEREAD,
        SPPHON_NONE,
        NONWORD_NONE,
        APTGRAM_GAIN,
        APTINFO_GAIN,
        B1EXNT_GAIN,
        B1EXTAU_GAIN,
        B1EXTO_GAIN,
        B1RENT_GAIN,
        B1RETAU_GAIN,
        B1RETO_GAIN,
        B2EXNT_GAIN,
        B2EXTAU_GAIN,
        B2EXTO_GAIN,
        B2RENT_GAIN,
        B2RETAU_GAIN,
        B2RETO_GAIN,
        BLENDING_GAIN,
        CELF_GAIN,
        DEAPPAV_GAIN,
        DEAPPFI_GAIN,
        DEAPPIN_GAIN,
        DEAPPVO_GAIN,
        EOWPVT_GAIN,
        ERBNW_GAIN,
        ERBTO_GAIN,
        ERBWORD_GAIN,
        EWRSWR_GAIN,
        LSAMINT_GAIN,
        LSAMMAX_GAIN,
        LSAMMLU_GAIN,
        LSAMTO_GAIN,
        LSAMUN_GAIN,
        NONWORD_GAIN,
        ROWPVT_GAIN,
        SPPHON_GAIN,
        SPRAW_GAIN,
        TROG_GAIN,
        YARCEWR_GAIN,
        YARCLET_GAIN,
        YARCSI_GAIN,
        APTGRAM_NEXT,
        APTINFO_NEXT,
        B1EXNT_NEXT,
        B1EXTAU_NEXT,
        B1EXTO_NEXT,
        B1RENT_NEXT,
        B1RETAU_NEXT,
        B1RETO_NEXT,
        B2EXNT_NEXT,
        B2EXTAU_NEXT,
        B2EXTO_NEXT,
        B2RENT_NEXT,
        B2RETAU_NEXT,
        B2RETO_NEXT,
        BLENDING_NEXT,
        CELF_NEXT,
        DEAPPAV_NEXT,
        DEAPPFI_NEXT,
        DEAPPIN_NEXT,
        DEAPPVO_NEXT,
        EOWPVT_NEXT,
        ERBNW_NEXT,
        ERBTO_NEXT,
        ERBWORD_NEXT,
        EWRSWR_NEXT,
        LSAMINT_NEXT,
        LSAMMAX_NEXT,
        LSAMMLU_NEXT,
        LSAMUN_NEXT,
        LSAMTO_NEXT,
        NONWORD_NEXT,
        ROWPVT_NEXT,
        SPPHON_NEXT,
        SPRAW_NEXT,
        TROG_NEXT,
        YARCEWR_NEXT,
        YARCLET_NEXT,
        YARCSI_NEXT,
    ]

    GAINS = [
        APTGRAM_GAIN,
        APTINFO_GAIN,
        B1EXNT_GAIN,
        B1EXTAU_GAIN,
        B1EXTO_GAIN,
        B1RENT_GAIN,
        B1RETAU_GAIN,
        B1RETO_GAIN,
        B2EXNT_GAIN,
        B2EXTAU_GAIN,
        B2EXTO_GAIN,
        B2RENT_GAIN,
        B2RETAU_GAIN,
        B2RETO_GAIN,
        BLENDING_GAIN,
        CELF_GAIN,
        DEAPPAV_GAIN,
        DEAPPFI_GAIN,
        DEAPPIN_GAIN,
        DEAPPVO_GAIN,
        EOWPVT_GAIN,
        ERBNW_GAIN,
        ERBTO_GAIN,
        ERBWORD_GAIN,
        EWRSWR_GAIN,
        LSAMINT_GAIN,
        LSAMMAX_GAIN,
        LSAMMLU_GAIN,
        LSAMUN_GAIN,
        LSAMTO_GAIN,
        NONWORD_GAIN,
        ROWPVT_GAIN,
        SPPHON_GAIN,
        SPRAW_GAIN,
        TROG_GAIN,
        YARCEWR_GAIN,
        YARCLET_GAIN,
        YARCSI_GAIN,
    ]

    NEXTS = [
        APTGRAM_NEXT,
        APTINFO_NEXT,
        B1EXNT_NEXT,
        B1EXTAU_NEXT,
        B1EXTO_NEXT,
        B1RENT_NEXT,
        B1RETAU_NEXT,
        B1RETO_NEXT,
        B2EXNT_NEXT,
        B2EXTAU_NEXT,
        B2EXTO_NEXT,
        B2RENT_NEXT,
        B2RETAU_NEXT,
        B2RETO_NEXT,
        BLENDING_NEXT,
        CELF_NEXT,
        DEAPPAV_NEXT,
        DEAPPFI_NEXT,
        DEAPPIN_NEXT,
        DEAPPVO_NEXT,
        EOWPVT_NEXT,
        ERBNW_NEXT,
        ERBTO_NEXT,
        ERBWORD_NEXT,
        EWRSWR_NEXT,
        LSAMINT_NEXT,
        LSAMMAX_NEXT,
        LSAMMLU_NEXT,
        LSAMTO_NEXT,
        LSAMUN_NEXT,
        NONWORD_NEXT,
        ROWPVT_NEXT,
        SPPHON_NEXT,
        SPRAW_NEXT,
        TROG_NEXT,
        YARCEWR_NEXT,
        YARCLET_NEXT,
        YARCSI_NEXT,
    ]

    NUMERIC = [
        AGE,
        BLOCKS,
        APTGRAM,
        APTINFO,
        B1EXTAU,
        B1EXTO,
        B1EXNT,
        B1RETAU,
        B1RETO,
        B1RENT,
        B2EXTAU,
        B2EXTO,
        B2EXNT,
        B2RETAU,
        B2RETO,
        B2RENT,
        CELF,
        EOWPVT,
        ERBNW,
        ERBTO,
        ERBWORD,
        LSAMMAX,
        LSAMMLU,
        LSAMINT,
        LSAMUN,
        LSAMTO,
        NONWORD,
        BLENDING,
        ROWPVT,
        SPPHON,
        SPRAW,
        TROG,
        YARCEWR,
        YARCLET,
        YARCSI,
        DEAPPIN,
        DEAPPVO,
        DEAPPFI,
        DEAPPAV,
        DEAPP_C,
        EWRSWR,
        ATTEND,
        ATTEND_CUMUL,
        TACHANG,
        SDQ,
        AGESPEAK,
        NUMCHIL,
        AGEBOOKS,
        MUMEDUPOST16,
        DADEDUPOST16,
    ]
    """
    Numeric variables in the dataset, not including gains and next time point variables.
    """

    CATEGORICAL = [
        TIME,
        GROUP,
        AREA,
        GENDER,
        VISION,
        HEARING,
        BEHAV,
        TASCORE,
        EARINF,
        HEARING_C,
        MUMOCC,
        DADOCC,
        BEDTIMEREAD,  # 7 = 7 or more
        OTHERTIMEREAD,  # 7 = 7 or more
        SPPHON_NONE,
        NONWORD_NONE,
    ]
    """
    Categorical variables in the dataset.
    """

    DEFAULT_EXCLUDED = [
        BLOCKS,  # only at t1
        B1EXTAU,  # included in B1EXTO
        B1RETAU,  # included in B1RETO
        B1EXNT,  # included in B1EXTO
        B1RENT,  # included in B1RETO
        B2RETAU,  # included in B2RETO
        B2RENT,  # included in B2RETO
        B2RETO,  # not recorded at t1
        B2EXTAU,  # included in B2EXTO
        B2EXNT,  # included in B2EXTO
        B2EXTO,  # not recorded at t1
        YARCEWR,  # included in EWRSWR
        ERBTO,  # includes ERBWORD and ERBNW
        DEAPPAV,  # averages DEAPPIN, DEAPPVO, DEAPPFI
        DEAPP_C,  # sums DEAPPIN, DEAPPVO, DEAPPFI
        MUMOCC,  # only at t1, not clear how coded
        DADOCC,  # only at t1, not clear how coded
        SPRAW,  # included in SPPHON
        TACHANG,  # only at t1-t2, t2-t3
        TASCORE,  # only at t1-t2, t2-t3
        SDQ,  # only at t1
        BEDTIMEREAD,  # only at t1
        OTHERTIMEREAD,  # only at t1,
        LSAMTO,  # only at t1, t2
        LSAMMAX,  # only at t1, t2
        LSAMMLU,  # only at t1, t2
        LSAMINT,  # only at t1, t2
        LSAMUN,  # only at t1, t2
        SPPHON_NONE,  # binary indicator of SPPHON < 1 at all time points
        NONWORD_NONE,  # binary indicator of NONWORD < 1 at all time points
        HEARING_C,  # combines HEARING and EARINF
    ]

    PERIOD_RELATED = [
        ATTEND,  # intervention sessions between time points
        TASCORE,  # TA rating between time points
        TACHANG,  # TA changes between time points
    ]

    TIME_INVARIANT_BASELINES = [
        HEARING,
        EARINF,
        HEARING_C,
        VISION,
        NUMCHIL,
        MUMOCC,
        DADOCC,
        AGEBOOKS,
        MUMEDUPOST16,
        DADEDUPOST16,
        BEDTIMEREAD,
        OTHERTIMEREAD,
        BLOCKS,
        AGESPEAK,
    ]
    """
    Variables captured once at study baseline (parent report or a single
    assessment at t1) and replicated across every timepoint in the
    long-format data.

    These features are time-invariant *within child*: each child
    contributes the same value across their 3–4 rows. Under
    ``GroupKFold`` grouping by ``subject_id`` there is no test-set
    leakage, but during training the model still sees each child's
    value repeated, which inflates effective training support and can
    bias tree splits and permutation importance toward these features.

    Treat this list as a *flag*, not a hard exclusion. It exists so
    that:

    - feature-selection review can identify which predictors in a
      final set are time-invariant and warrant an extra sensitivity
      check (drop-and-retune);
    - level-model reports can surface the time-invariance of a
      predictor inline alongside its importance;
    - downstream pipeline options (e.g. inverse-frequency subject
      weighting) can consume a single canonical list rather than
      re-deriving it per model.

    The concern is structurally weaker for gain models (where
    time-invariant features can only explain between-child *trajectory*
    differences, not within-child change) than for level models.
    """

    NAMES = {
        SUBJECT_ID: "Subject ID",
        TIME: "Time",
        GROUP: "Group",
        AREA: "Area",
        GENDER: "Gender",
        AGE: "Age",
        BLOCKS: "Block design score",
        APTGRAM: "APT expressive grammar score",
        APTINFO: "APT expressive information score",
        B1EXTAU: "Block 1 taught expressive vocabulary score",
        B1EXTO: "Block 1 total expressive vocabulary score",
        B1EXNT: "Block 1 not taught expressive vocabulary score",
        CELF: "Basic concept knowledge score",
        EOWPVT: "Expressive vocabulary score",
        ROWPVT: "Receptive vocabulary score",
        TROG: "Receptive grammar score",
        ERBNW: "Nonword repitition score",
        ERBTO: "Word and nonword repetition score",
        ERBWORD: "Word repetition score",
        NONWORD: "Nonword reading score",
        BLENDING: "Phoneme blending score",
        SPPHON: "Phonetic spelling score",
        SPRAW: "Word spelling score",
        YARCEWR: "Early word recognition score",
        YARCLET: "Letter sounds score",
        YARCSI: "Comprehension skills score",
        EWRSWR: "Early word reading composite score",
        BEHAV: "Behaviour rating",
        ATTEND: "Intervention sessions attended",
    }

    @staticmethod
    def get_variable_name(var: str) -> str:
        n = Variables.NAMES.get(var)
        return n if n else var


class Predictors:

    DEFAULT_GAIN = [
        v
        for v in Variables.ALL
        if v != Variables.SUBJECT_ID
        and v not in Variables.DEFAULT_EXCLUDED
        and v not in Variables.GAINS
        and v not in Variables.NEXTS
    ]
    """
    The default predictor variables for gain outcomes are all measure variables (excluding 
    subject ID, gain variables and variables included in composites or only measured at t1).
    """

    DEFAULT_GAIN_NUMERIC = [v for v in DEFAULT_GAIN if v in Variables.NUMERIC]
    """
    All DEFAULT_GAIN predictor variables that are numeric.
    """

    DEFAULT_GAIN_CATEGORICAL = [v for v in DEFAULT_GAIN if v in Variables.CATEGORICAL]
    """
    All DEFAULT_GAIN variables that are categorical.
    """

    DEFAULT_LEVEL = [
        v
        for v in Variables.ALL
        if v != Variables.SUBJECT_ID
        and v not in Variables.DEFAULT_EXCLUDED
        and v not in Variables.GAINS
        and v not in Variables.NEXTS
        and v not in Variables.PERIOD_RELATED
    ]
    """
    The default predictor variables for level outcomes are all measure variables (excluding 
    subject ID, period-related measures (attendance, TA score) gain variables and variables 
    included in composites or only measured at t1).
    """

    DEFAULT_LEVEL_NUMERIC = [v for v in DEFAULT_LEVEL if v in Variables.NUMERIC]
    """
    All DEFAULT_LEVEL predictor variables that are numeric.
    """

    DEFAULT_LEVEL_CATEGORICAL = [v for v in DEFAULT_LEVEL if v in Variables.CATEGORICAL]
    """
    All DEFAULT_LEVEL variables that are categorical.
    """


class Categories:
    """
    Categories for variables and variable values.
    """

    GENDER = {
        1: "Male",
        2: "Female",
    }

    AREA = {
        1: "North",
        2: "South",
    }

    GROUP = {
        1: "Initial intervention",
        2: "Wait for intervention",
    }

    TIME = {
        1: "Time 1",
        2: "Time 2",
        3: "Time 3",
        4: "Time 4",
    }

    TIME_PERIOD = {
        1: "Period 1",
        2: "Period 2",
        3: "Period 3",
    }

    BEHAVIOUR = {
        1: "Very good",
        2: "Good",
        3: "Average",
        4: "Challenging",
        5: "Very challenging",
    }

    IMPAIRED = {
        0: "Normal",
        1: "Impaired",
    }
    """
    Binary indicator of impairment - for VISION and HEARING variables.
    """

    NO_YES = {
        0: "No",
        1: "Yes",
    }

    WEEKLY_READING = {
        0: "None",
        1: "Once a week",
        2: "Twice a week",
        3: "Three times a week",
        4: "Four times a week",
        5: "Five times a week",
        6: "Six times a week",
        7: "Seven or more times a week",
    }

    NON_SCORER = {
        0: "Scored > 0 at least once",
        1: "Never scored > 0",
    }

    VARIABLES = {
        0: "Study",
        1: "Health",
        2: "Child and Family",
        3: "Cognition",
        4: "Language",
        5: "Speech",
        6: "Reading",
    }
    """
    Dictionary mapping variable category IDs to category names.
    """

    STUDY = [Variables.TIME, Variables.GROUP, Variables.AREA]
    """
    Study-related variables.
    """

    HEALTH = [
        Variables.HEARING,
        Variables.EARINF,
        Variables.HEARING_C,
        Variables.VISION,
    ]
    """
    Health-related variables.
    """

    CHILD_FAMILY = [
        Variables.AGE,
        Variables.NUMCHIL,
        Variables.MUMOCC,
        Variables.DADOCC,
        Variables.AGEBOOKS,
        Variables.MUMEDUPOST16,
        Variables.DADEDUPOST16,
        Variables.BEDTIMEREAD,
        Variables.OTHERTIMEREAD,
    ]
    """
    Child and family-related variables.
    """

    COGNITION = [
        Variables.BLOCKS,
    ]

    LANGUAGE = [
        Variables.APTGRAM,
        Variables.APTINFO,
        Variables.ROWPVT,
        Variables.EOWPVT,
        Variables.B1EXTAU,
        Variables.B1EXTO,
        Variables.B1EXNT,
        Variables.B1RETAU,
        Variables.B1RETO,
        Variables.B1RENT,
        Variables.B2EXTAU,
        Variables.B2EXTO,
        Variables.B2EXNT,
        Variables.B2RETAU,
        Variables.B2RETO,
        Variables.B2RENT,
        Variables.CELF,
        Variables.TROG,
        Variables.ROWPVT_NEXT,
        Variables.EOWPVT_NEXT,
        Variables.TROG_NEXT,
        Variables.CELF_NEXT,
        Variables.APTGRAM_NEXT,
        Variables.APTINFO_NEXT,
        Variables.B1EXNT_NEXT,
        Variables.B1EXTO_NEXT,
        Variables.B1EXTAU_NEXT,
        Variables.B1RENT_NEXT,
        Variables.B1RETO_NEXT,
        Variables.B1RETAU_NEXT,
        Variables.B2EXNT_NEXT,
        Variables.B2EXTO_NEXT,
        Variables.B2EXTAU_NEXT,
        Variables.B2RENT_NEXT,
        Variables.B2RETO_NEXT,
        Variables.B2RETAU_NEXT,
        Variables.ROWPVT_GAIN,
        Variables.EOWPVT_GAIN,
        Variables.TROG_GAIN,
        Variables.CELF_GAIN,
        Variables.APTGRAM_GAIN,
        Variables.APTINFO_GAIN,
        Variables.B1EXNT_GAIN,
        Variables.B1EXTO_GAIN,
        Variables.B1EXTAU_GAIN,
        Variables.B1RENT_GAIN,
        Variables.B1RETO_GAIN,
        Variables.B1RETAU_GAIN,
        Variables.B2EXNT_GAIN,
        Variables.B2EXTO_GAIN,
        Variables.B2EXTAU_GAIN,
        Variables.B2RENT_GAIN,
        Variables.B2RETO_GAIN,
        Variables.B2RETAU_GAIN,
    ]

    SPEECH = [
        Variables.ERBNW,
        Variables.ERBTO,
        Variables.ERBWORD,
        Variables.LSAMMAX,
        Variables.LSAMMLU,
        Variables.LSAMINT,
        Variables.LSAMUN,
        Variables.LSAMTO,
        Variables.DEAPPIN,
        Variables.DEAPPVO,
        Variables.DEAPPFI,
        Variables.DEAPPAV,
        Variables.DEAPP_C,
        Variables.AGESPEAK,
        Variables.ERBWORD_NEXT,
        Variables.ERBNW_NEXT,
        Variables.ERBTO_NEXT,
        Variables.DEAPPIN_NEXT,
        Variables.DEAPPVO_NEXT,
        Variables.DEAPPFI_NEXT,
        Variables.DEAPPAV_NEXT,
        Variables.LSAMMAX_NEXT,
        Variables.LSAMMLU_NEXT,
        Variables.LSAMINT_NEXT,
        Variables.LSAMTO_NEXT,
        Variables.LSAMUN_NEXT,
        Variables.ERBWORD_GAIN,
        Variables.ERBNW_GAIN,
        Variables.ERBTO_GAIN,
        Variables.DEAPPIN_GAIN,
        Variables.DEAPPVO_GAIN,
        Variables.DEAPPFI_GAIN,
        Variables.DEAPPAV_GAIN,
        Variables.LSAMMAX_GAIN,
        Variables.LSAMMLU_GAIN,
        Variables.LSAMINT_GAIN,
        Variables.LSAMTO_GAIN,
        Variables.LSAMUN_GAIN,
    ]

    READING = [
        Variables.EWRSWR,
        Variables.NONWORD,
        Variables.BLENDING,  # ? only spoken words and pictures
        Variables.SPPHON,
        Variables.SPRAW,
        Variables.YARCEWR,
        Variables.YARCLET,
        Variables.YARCSI,
        Variables.SPPHON_NONE,
        Variables.NONWORD_NONE,
        Variables.EWRSWR_NEXT,
        Variables.YARCEWR_NEXT,
        Variables.YARCLET_NEXT,
        Variables.YARCSI_NEXT,
        Variables.SPPHON_NEXT,
        Variables.SPRAW_NEXT,
        Variables.NONWORD_NEXT,
        Variables.BLENDING_NEXT,
        Variables.EWRSWR_GAIN,
        Variables.YARCEWR_GAIN,
        Variables.YARCLET_GAIN,
        Variables.YARCSI_GAIN,
        Variables.SPPHON_GAIN,
        Variables.SPRAW_GAIN,
        Variables.NONWORD_GAIN,
        Variables.BLENDING_GAIN,
    ]

    SOCIAL = [
        Variables.BEHAV,
        Variables.SDQ,
    ]

    TEACHING = [
        Variables.ATTEND,
        Variables.ATTEND_CUMUL,
        Variables.TASCORE,
        Variables.TACHANG,
    ]
