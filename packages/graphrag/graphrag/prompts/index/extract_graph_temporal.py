# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Temporal-aware graph extraction prompt definitions.

Extends the standard graph extraction prompt with a temporal_scope field
for entities and relationships, allowing the LLM to capture when facts
apply (e.g. "2020-2023", "since January 2024", "formerly").
"""

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities. For each extraction, also provide any temporal context if mentioned.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
- temporal_scope: If mentioned in the text, the time period this entity description applies to (e.g. "2020-2023", "since January 2024", "formerly", "as of Q3 2024", "until March 2025"). Leave empty if no temporal context is apparent in the text.
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description><|><temporal_scope>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- temporal_scope: If mentioned, the time period this relationship applies to. Leave empty if no temporal context.
 Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength><|><temporal_scope>)

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
("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday<|>)
##
("entity"<|>MARTIN SMITH<|>PERSON<|>Martin Smith is the chair of the Central Institution<|>)
##
("entity"<|>MARKET STRATEGY COMMITTEE<|>ORGANIZATION<|>The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply<|>)
##
("relationship"<|>MARTIN SMITH<|>CENTRAL INSTITUTION<|>Martin Smith is the Chair of the Central Institution and will answer questions at a press conference<|>9<|>)
<|COMPLETE|>

######################
Example 2:
Entity_types: ORGANIZATION
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"<|>TECHGLOBAL<|>ORGANIZATION<|>TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones<|>)
##
("entity"<|>VISION HOLDINGS<|>ORGANIZATION<|>Vision Holdings is a firm that previously owned TechGlobal<|>)
##
("relationship"<|>TECHGLOBAL<|>VISION HOLDINGS<|>Vision Holdings took TechGlobal private<|>5<|>2014)
##
("relationship"<|>VISION HOLDINGS<|>TECHGLOBAL<|>Vision Holdings formerly owned TechGlobal<|>5<|>2014 until present)
<|COMPLETE|>

######################
Example 3:
Entity_types: ORGANIZATION,GEO,PERSON
Text:
In January 2024, Acme Corp appointed Jane Doe as CEO, replacing long-time leader Bob Wilson who had served since 2015. Under Wilson's leadership from 2015 to 2023, the company expanded into three new markets. Jane Doe previously served as CTO of Widgets Inc from 2019 to 2023.
######################
Output:
("entity"<|>ACME CORP<|>ORGANIZATION<|>Acme Corp is a company that expanded into three new markets under prior leadership<|>)
##
("entity"<|>JANE DOE<|>PERSON<|>Jane Doe is the CEO of Acme Corp, previously CTO of Widgets Inc<|>since January 2024)
##
("entity"<|>BOB WILSON<|>PERSON<|>Bob Wilson is the former CEO of Acme Corp who oversaw expansion into three new markets<|>2015-2023)
##
("entity"<|>WIDGETS INC<|>ORGANIZATION<|>Widgets Inc is a company where Jane Doe previously served as CTO<|>)
##
("relationship"<|>JANE DOE<|>ACME CORP<|>Jane Doe was appointed CEO of Acme Corp<|>9<|>since January 2024)
##
("relationship"<|>BOB WILSON<|>ACME CORP<|>Bob Wilson served as CEO of Acme Corp and led expansion into three new markets<|>9<|>2015-2023)
##
("relationship"<|>JANE DOE<|>WIDGETS INC<|>Jane Doe served as CTO of Widgets Inc<|>7<|>2019-2023)
##
("relationship"<|>JANE DOE<|>BOB WILSON<|>Jane Doe replaced Bob Wilson as CEO of Acme Corp<|>8<|>January 2024)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format (including temporal_scope where applicable):\n"
LOOP_PROMPT = "It appears some entities and relationships may have still been missed. Answer Y if there are still entities or relationships that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"
