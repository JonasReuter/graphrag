# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Evidence-enhanced graph extraction prompt definitions."""

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities. For each extraction, also provide a confidence score, completeness assessment, and the exact source quote.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
- confidence: A float between 0.0 and 1.0 indicating how confident you are in this extraction. Use 1.0 for explicit, unambiguous mentions. Use lower values for implicit or ambiguous references.
- completeness: One of "complete", "partial", or "inferred". Use "complete" if all key attributes are captured. Use "partial" if important attributes like time period, role, or conditions are missing. Use "inferred" if the entity is implied but not explicitly stated.
- source_quote: The exact text span from the input that supports this extraction. Keep it concise but sufficient to verify the claim.
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description><|><confidence><|><completeness><|><source_quote>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- confidence: A float between 0.0 and 1.0 indicating how confident you are in this relationship. Use 1.0 for explicitly stated relationships. Use lower values for implied or uncertain connections.
- completeness: One of "complete", "partial", or "inferred". Use "partial" if important qualifiers like time period, conditions, or role are missing. Use "inferred" if the relationship is implied but not directly stated.
- source_quote: The exact text span from the input that supports this relationship.
 Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength><|><confidence><|><completeness><|><source_quote>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday<|>1.0<|>complete<|>The Verdantis's Central Institution is scheduled to meet on Monday and Thursday)
##
("entity"<|>MARTIN SMITH<|>PERSON<|>Martin Smith is the chair of the Central Institution<|>1.0<|>complete<|>Central Institution Chair Martin Smith will take questions)
##
("entity"<|>MARKET STRATEGY COMMITTEE<|>ORGANIZATION<|>The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply<|>0.9<|>partial<|>Investors expect the Market Strategy Committee to hold its benchmark interest rate steady)
##
("relationship"<|>MARTIN SMITH<|>CENTRAL INSTITUTION<|>Martin Smith is the Chair of the Central Institution and will answer questions at a press conference<|>9<|>1.0<|>complete<|>Central Institution Chair Martin Smith will take questions)
<|COMPLETE|>

######################
Example 2:
Entity_types: ORGANIZATION
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"<|>TECHGLOBAL<|>ORGANIZATION<|>TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones<|>1.0<|>complete<|>TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday)
##
("entity"<|>VISION HOLDINGS<|>ORGANIZATION<|>Vision Holdings is a firm that previously owned TechGlobal<|>1.0<|>complete<|>was taken private by Vision Holdings in 2014)
##
("relationship"<|>TECHGLOBAL<|>VISION HOLDINGS<|>Vision Holdings formerly owned TechGlobal from 2014 until present<|>5<|>0.8<|>partial<|>was taken private by Vision Holdings in 2014)
<|COMPLETE|>

######################
Example 3:
Entity_types: ORGANIZATION,GEO,PERSON
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
("entity"<|>FIRUZABAD<|>GEO<|>Firuzabad held Aurelians as hostages<|>1.0<|>complete<|>Five Aurelians jailed for 8 years in Firuzabad)
##
("entity"<|>AURELIA<|>GEO<|>Country seeking to release hostages<|>1.0<|>complete<|>are on their way home to Aurelia)
##
("entity"<|>QUINTARA<|>GEO<|>Country that negotiated a swap of money in exchange for hostages<|>1.0<|>complete<|>The swap orchestrated by Quintara)
##
("entity"<|>TIRUZIA<|>GEO<|>Capital of Firuzabad where the Aurelians were being held<|>1.0<|>complete<|>Firuzabad's capital, Tiruzia)
##
("entity"<|>KROHAARA<|>GEO<|>Capital city in Quintara<|>1.0<|>complete<|>financial institutions in Krohaara, the capital of Quintara)
##
("entity"<|>CASHION<|>GEO<|>Capital city in Aurelia<|>0.9<|>complete<|>Aurelia's capital, Cashion)
##
("entity"<|>SAMUEL NAMARA<|>PERSON<|>Aurelian businessman, 39 years old, who spent time in Tiruzia's Alhamia Prison<|>1.0<|>complete<|>39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison)
##
("entity"<|>ALHAMIA PRISON<|>GEO<|>Prison in Tiruzia<|>1.0<|>complete<|>Tiruzia's Alhamia Prison)
##
("entity"<|>DURKE BATAGLANI<|>PERSON<|>Aurelian journalist, 59 years old, who was held hostage<|>1.0<|>complete<|>journalist Durke Bataglani, 59)
##
("entity"<|>MEGGIE TAZBAH<|>PERSON<|>Bratinas national and environmentalist, 53 years old, who was held hostage<|>1.0<|>complete<|>environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality)
##
("relationship"<|>FIRUZABAD<|>AURELIA<|>Firuzabad negotiated a hostage exchange with Aurelia<|>2<|>0.9<|>inferred<|>Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia)
##
("relationship"<|>QUINTARA<|>AURELIA<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2<|>1.0<|>complete<|>The swap orchestrated by Quintara)
##
("relationship"<|>QUINTARA<|>FIRUZABAD<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2<|>1.0<|>complete<|>The swap orchestrated by Quintara)
##
("relationship"<|>SAMUEL NAMARA<|>ALHAMIA PRISON<|>Samuel Namara was a prisoner at Alhamia prison<|>8<|>1.0<|>complete<|>Samuel Namara, who has been held in Tiruzia's Alhamia Prison)
##
("relationship"<|>SAMUEL NAMARA<|>MEGGIE TAZBAH<|>Samuel Namara and Meggie Tazbah were exchanged in the same hostage release<|>2<|>0.9<|>inferred<|>The Aurelians include 39-year-old businessman Samuel Namara...as well as...Meggie Tazbah)
##
("relationship"<|>SAMUEL NAMARA<|>DURKE BATAGLANI<|>Samuel Namara and Durke Bataglani were exchanged in the same hostage release<|>2<|>0.9<|>inferred<|>The Aurelians include 39-year-old businessman Samuel Namara, as well as journalist Durke Bataglani)
##
("relationship"<|>MEGGIE TAZBAH<|>DURKE BATAGLANI<|>Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release<|>2<|>0.9<|>inferred<|>journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53)
##
("relationship"<|>SAMUEL NAMARA<|>FIRUZABAD<|>Samuel Namara was a hostage in Firuzabad<|>2<|>1.0<|>complete<|>Samuel Namara, who has been held in Tiruzia's Alhamia Prison)
##
("relationship"<|>MEGGIE TAZBAH<|>FIRUZABAD<|>Meggie Tazbah was a hostage in Firuzabad<|>2<|>0.8<|>inferred<|>Five Aurelians jailed for 8 years in Firuzabad)
##
("relationship"<|>DURKE BATAGLANI<|>FIRUZABAD<|>Durke Bataglani was a hostage in Firuzabad<|>2<|>0.8<|>inferred<|>Five Aurelians jailed for 8 years in Firuzabad)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format (including confidence, completeness, and source_quote):\n"
LOOP_PROMPT = "It appears some entities and relationships may have still been missed. Answer Y if there are still entities or relationships that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"
