import copy
from datetime import datetime, timezone
import json
from operator import itemgetter
import xml.etree.ElementTree as ElTree
from download import query_pubtator2biocxml


def pubtator2dict_list(pubtator_file_path, is_raw_text):
    dict_list = list()

    title_pmid = ''
    # abstract_pmid = ''
    title = ''
    abstract_text = ''
    doc_line_num = 0
    mutations = list()

    with open(pubtator_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                # temp
                # if title_pmid != abstract_pmid:
                #     return '{"error": "pmid disagreement {} != {}"}'\
                #         .format(title_pmid, abstract_pmid)

                if len(mutations) > 0:
                    if len(mutations) > 1:
                        mutations = sorted(mutations,
                                           key=itemgetter('start'))

                    if is_raw_text:
                        # abstract only
                        mutations = get_bestplus_spans(mutations, abstract_text)
                    else:
                        # a title, a space and an abstract
                        mutations = get_bestplus_spans(mutations,
                                                       title + ' ' +
                                                       abstract_text)
                    # print('Found mutation(s)', mutations)

                doc_dict = {
                    'pmid': title_pmid,
                    'mutation_model': 'tmVar 2.0',
                    'entities': {'mutation': copy.deepcopy(mutations)}
                }
                if is_raw_text:
                    doc_dict['abstract'] = abstract_text
                else:
                    doc_dict['title'] = title
                    doc_dict['abstract'] = abstract_text

                dict_list.append(doc_dict)

                doc_line_num = 0
                mutations.clear()
                continue

            if doc_line_num == 0:
                title_cols = line.split('|t|')

                if len(title_cols) != 2:
                    return '{"error": "wrong #title_cols {}"}'\
                        .format(len(title_cols))

                title_pmid = title_cols[0]

                if '- No text -' == title_cols[1]:
                    # make tmvar2 results empty
                    title = ''
                else:
                    title = title_cols[1]
            elif doc_line_num == 1:
                abstract_cols = line.split('|a|')

                if len(abstract_cols) != 2:
                    return '{"error": "wrong #abstract_cols {}"}' \
                        .format(len(abstract_cols))

                if '- No text -' == abstract_cols[1]:
                    # make tmvar2 results empty
                    abstract_text = ''
                else:
                    abstract_text = abstract_cols[1]
            elif doc_line_num > 1:
                mutation_cols = line.split('\t')

                if len(mutation_cols) != 6:
                    return '{"error": "wrong #mutation_cols {}"}' \
                        .format(len(mutation_cols))

                mutations.append({'start': int(mutation_cols[1]),
                                  'end': int(mutation_cols[2]),
                                  'mention': mutation_cols[3],
                                  'mutationType': mutation_cols[4],
                                  'normalizedName': mutation_cols[5]})

            doc_line_num += 1
    return dict_list


def pubtatorstr2dict_list(pubtator, is_raw_text):
    mutation_types = ['ProteinMutation', 'DNAMutation', 'SNP']

    dict_list = list()

    title_pmid = ''
    abstract_pmid = ''
    title = ''
    abstract_text = ''
    doc_line_num = 0
    mutations = list()

    for line in pubtator.splitlines():
        if len(line) == 0:

            if title_pmid != abstract_pmid:
                return '{"error": "pmid disagreement"}'

            if len(mutations) > 0:
                if len(mutations) > 1:
                    mutations = sorted(mutations,
                                       key=itemgetter('start'))

                if is_raw_text:
                    # tmtool: title only
                    mutations = get_bestplus_spans(mutations, title)
                else:
                    # a title, a space and an abstract
                    mutations = get_bestplus_spans(mutations,
                                                   title + ' ' +
                                                   abstract_text)
                # print('Found mutation(s)', mutations)

            doc_dict = {
                'pmid': title_pmid,
                'mutation_model': 'tmtool tmVar',
                'entities': {'mutation': copy.deepcopy(mutations)}
            }
            if is_raw_text:
                # tmtool: title only & STM
                doc_dict['abstract'] = title
            else:
                doc_dict['title'] = title
                doc_dict['abstract'] = abstract_text

            dict_list.append(doc_dict)

            doc_line_num = 0
            mutations.clear()
            continue

        if doc_line_num == 0:
            title_cols = line.split('|t|')

            if len(title_cols) != 2:
                return '{"error": "wrong #title_cols {}"}' \
                    .format(len(title_cols))

            title_pmid = title_cols[0]

            if '- No text -' == title_cols[1]:
                # make tmvar2 results empty
                title = ''
            else:
                title = title_cols[1]
        elif doc_line_num == 1:
            abstract_cols = line.split('|a|')

            if len(abstract_cols) != 2:
                return '{"error": "wrong #abstract_cols {}"}' \
                    .format(len(abstract_cols))

            abstract_pmid = abstract_cols[0]

            if '- No text -' == abstract_cols[1] \
                    or '-NoAbstract-' == abstract_cols[1]:
                # make tmvar2 results empty
                abstract_text = ''
            else:
                abstract_text = abstract_cols[1]
        elif doc_line_num > 1:
            mutation_cols = line.split('\t')

            if len(mutation_cols) != 6:
                return '{"error": "wrong #mutation_cols {}"}' \
                    .format(len(mutation_cols))

            if mutation_cols[4] in mutation_types:
                mutations.append({'start': int(mutation_cols[1]),
                                  'end': int(mutation_cols[2]),
                                  'mention': mutation_cols[3],
                                  'mutationType': mutation_cols[4],
                                  'normalizedName': mutation_cols[5]})

        doc_line_num += 1
    return dict_list


def pubtator_biocxml2dict_list(pmids):
    pubtator_xml, pubtator_xml_raw = query_pubtator2biocxml(pmids)

    if type(pubtator_xml) is not str:
        return None

    if 'document' not in pubtator_xml and 'passage' not in pubtator_xml:
        return pubtator_xml

    mutation_types = ['ProteinMutation', 'DNAMutation', 'SNP']

    temp_dict = dict()

    try:
        root = ElTree.fromstring(pubtator_xml)
    except ElTree.ParseError as pe:
        print('XML ParseError', pe.msg)
        root = ElTree.fromstring(pubtator_xml_raw)

    for child in root:

        if 'document' != child.tag:
            continue

        pubmed_id = child.find('id').text

        text_dict = {
            'title': '',
            'abstract': ''
        }

        mutations = list()

        for passage in child.iter('passage'):

            passage_type = passage.find(".//*[@key='type']").text

            for p_child in passage:
                if 'text' == p_child.tag:
                    text_dict[passage_type] = p_child.text
                elif 'annotation' == p_child.tag:
                    entity_type = p_child.find(".//*[@key='type']").text

                    if entity_type in mutation_types:
                        location = p_child.find('location')
                        start_offset = int(location.get('offset'))
                        end_offset = start_offset + int(location.get('length'))
                        mention = preprocess(p_child.find('text').text)
                        identifier = p_child.findall('infon')[0].text

                        mutations.append({'start': int(location.get('offset')),
                                          'end': end_offset,
                                          'mention': mention,
                                          'mutationType': entity_type,
                                          'normalizedName': identifier})

        if len(mutations) > 1:
            mutations = sorted(mutations, key=itemgetter('start'))

        title = preprocess(text_dict['title'])
        abstr = preprocess(text_dict['abstract'])

        # a title, a space and an abstract
        if len(abstr) > 0:
            mutations = get_bestplus_spans(mutations, title + ' ' + abstr)
        else:
            mutations = get_bestplus_spans(mutations, title)

        doc_dict = {
            'pmid': pubmed_id,
            'mutation_model': 'tmtool tmVar biocxml',
            'entities': {'mutation': mutations},
            'title': title,
            'abstract': abstr
        }

        temp_dict[pubmed_id] = doc_dict

    dict_list = list()
    empty_list = list()
    for pmid in pmids:
        if type(pmid) is int:
            pmid = str(pmid)

        if pmid not in temp_dict:
            dict_list.append(
                {
                    'pmid': pmid,
                    'mutation_model': 'tmtool tmVar biocxml',
                    'entities': {'mutation': empty_list},
                    'title': '',
                    'abstract': ''
                }
            )
        else:
            dict_list.append(temp_dict[pmid])

    return dict_list


def preprocess(text):
    text = text.replace('\r ', ' ')

    text = text.replace('\u2028', ' ')
    text = text.replace('\u2029', ' ')

    # HAIR SPACE
    # https://www.fileformat.info/info/unicode/char/200a/index.htm
    text = text.replace('\u200A', ' ')

    # THIN SPACE
    # https://www.fileformat.info/info/unicode/char/2009/index.htm
    text = text.replace('\u2009', ' ')
    text = text.replace('\u2008', ' ')

    # FOUR-PER-EM SPACE
    # https://www.fileformat.info/info/unicode/char/2005/index.htm
    text = text.replace('\u2005', ' ')
    text = text.replace('\u2004', ' ')
    text = text.replace('\u2003', ' ')

    # EN SPACE
    # https://www.fileformat.info/info/unicode/char/2002/index.htm
    text = text.replace('\u2002', ' ')

    # NO-BREAK SPACE
    # https://www.fileformat.info/info/unicode/char/00a0/index.htm
    text = text.replace('\u00A0', ' ')

    # https://www.fileformat.info/info/unicode/char/f8ff/index.htm
    text = text.replace('\uF8FF', ' ')

    # https://www.fileformat.info/info/unicode/char/202f/index.htm
    text = text.replace('\u202F', ' ')

    text = text.replace('\uFEFF', ' ')
    text = text.replace('\uF044', ' ')
    text = text.replace('\uF02D', ' ')
    text = text.replace('\uF0BB', ' ')

    text = text.replace('\uF048', '??')
    text = text.replace('\uF0B0', '??')

    # MIDLINE HORIZONTAL ELLIPSIS: ???
    # https://www.fileformat.info/info/unicode/char/22ef/index.htm
    # text = text.replace('\u22EF', '...')

    return text


def pubtator2pubannotation(pubtator):
    dict_list = list()

    title_pmid = ''
    abstract_pmid = ''
    title = ''
    abstract_text = ''
    doc_line_num = 0
    entities = list()

    for line in pubtator.splitlines():
        if len(line) == 0:

            if title_pmid != abstract_pmid:
                return '{"error": "pmid disagreement"}'

            doc_dict = {
                'project': 'BERN',
                'sourcedb': 'PubMed',
                'sourceid': title_pmid,
                'denotations': copy.deepcopy(entities),
                'text': title + ' ' + abstract_text
            }

            dict_list.append(doc_dict)

            doc_line_num = 0
            entities.clear()
            continue

        if doc_line_num == 0:
            title_cols = line.split('|t|')

            if len(title_cols) != 2:
                return '{"error": "wrong #title_cols=%d", "line": "%s"}' \
                       % (len(title_cols), line)

            title_pmid = title_cols[0]

            if '- No text -' == title_cols[1]:
                # make tmvar2 results empty
                title = ''
            else:
                title = title_cols[1]
        elif doc_line_num == 1:
            abstract_cols = line.split('|a|')

            if len(abstract_cols) != 2:
                return \
                    '{"error": "wrong #abstract_cols %d"}' % len(abstract_cols)

            abstract_pmid = abstract_cols[0]

            if '- No text -' == abstract_cols[1] \
                    or '-NoAbstract-' == abstract_cols[1]:
                # make tmvar2 results empty
                abstract_text = ''
            else:
                abstract_text = abstract_cols[1]
        elif doc_line_num > 1:
            entity_cols = line.split('\t')

            if len(entity_cols) != 6:
                return '{"error": "wrong #mutation_cols %d"}' % len(entity_cols)

            if entity_cols[4] in entity_cols:
                entities.append({'obj': entity_cols[4],
                                 'id': entity_cols[5].split('|'),
                                 'span': {
                                     'begin': int(entity_cols[1]),
                                     'end': int(entity_cols[2])
                                 }})

        doc_line_num += 1
    return dict_list


def get_bestplus_spans(mutations, title_space_abstract):
    adjusted_mutations = list()

    mention_count_dict = dict()
    for m in mutations:

        if 'No text' in m['mention']:
            continue

        if m['mention'] in mention_count_dict:
            mention_count_dict[m['mention']] += 1
        else:
            mention_count_dict[m['mention']] = 1

        count = mention_count_dict[m['mention']]

        start = -1
        found = 0
        while found < count:
            start = title_space_abstract.index(m['mention'], start + 1)
            assert start > -1
            found += 1

        end = start + len(m['mention']) - 1  # 2018.8.29 @chanho feedback

        assert m['mention'] == title_space_abstract[start: end + 1]

        adjusted_mutations.append({'start': start,
                                   'end': end,
                                   'mention': m['mention'],
                                   'mutationType': m['mutationType'],
                                   'normalizedName': m['normalizedName']})

    return adjusted_mutations


# Ref.
# http://pubannotation.org/docs/sourcedb/PubMed/sourceid/10022882/spans/606-710/annotations.json
# http://www.pubannotation.org/docs/annotation-format/
def get_pub_annotation(bern_dict, is_raw_text, elapsed_time_dict=None):
    sourceid = bern_dict['pmid']

    if is_raw_text:
        sourcedb = ''
        text = bern_dict['abstract']
    else:
        sourcedb = 'PubMed'
        if len(bern_dict['abstract']) > 0:
            if len(bern_dict['title']) > 0:
                text = bern_dict['title'] + ' ' + bern_dict['abstract']
            else:
                text = bern_dict['abstract']
        else:
            text = bern_dict['title']

    pa_dict = {
        'project': 'BERN',
        # 'target': '',
        'sourcedb': sourcedb,
        'sourceid': sourceid,
        'text': text,
        'denotations': bern2pub_annotation(bern_dict['entities']),
        # 'tracks': [{
        #     'project': 'BERN',
        #     'denotations': bern2pub_annotation(bern_dict['entities'])
        # }]
        'timestamp': datetime.now(tz=timezone.utc).strftime(
            '%a %b %d %H:%M:%S %z %Y')
    }

    if 'logits' in bern_dict:
        pa_dict['logits'] = bern_dict['logits']

    if elapsed_time_dict is not None:
        pa_dict['elapsed_time'] = elapsed_time_dict

    return pa_dict


def bern2pub_annotation(entity_dict):
    entity_list = list()
    for etype in entity_dict:
        for entity in entity_dict[etype]:

            # TODO prevention in the previous step
            if 'id' not in entity:
                entity['id'] = ['CUI-less']

            assert 'id' in entity, \
                '{}, entity={}, entity_dict={}'.format(
                    etype, entity, entity_dict)
            assert 'start' in entity and 'end' in entity, \
                '{}, entity={}, entity_dict={}'.format(
                    etype, entity, entity_dict)

            if '\t' in entity['id']:
                eid = entity['id'].split('\t')
            else:
                eid = [entity['id']]

            entity_pa_dict = {
                'id': eid,
                'span': {
                    'begin': entity['start'],
                    'end': entity['end']
                },
                'obj': etype
            }

            if 'mutation' == etype:

                assert 'mutationType' in entity \
                       and 'normalizedName' in entity, \
                    '{}, entity={}, entity_dict={}'.format(
                        etype, entity, entity_dict)

                entity_pa_dict['mutationType'] = entity['mutationType']
                entity_pa_dict['normalizedName'] = entity['normalizedName']

            entity_list.append(entity_pa_dict)

    # sort by span begin
    def get_item_key1(item):
        return item['span']['begin']

    def get_item_key2(item):
        return item['obj']

    return sorted(sorted(entity_list, key=get_item_key2), key=get_item_key1)


def get_pubtator(bern_dict_list):
    result = ''
    for bd in bern_dict_list:
        text = bd['title'] + ' ' + bd['abstract']

        main = bd['pmid'] + '|t|' + bd['title'] + '\n' + \
            bd['pmid'] + '|a|' + bd['abstract']

        # sort by start
        sorted_entities = list()

        for etype in bd['entities']:
            for entity in bd['entities'][etype]:
                mention = text[entity['start']: entity['end']]
                sorted_entities.append(
                    [entity['start'], entity['end'], mention, etype,
                     '|'.join(entity['id'].split('\t'))])

        sorted_entities = sorted(sorted_entities, key=itemgetter(0))

        entities = ''
        for e in sorted_entities:
            entities += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                bd['pmid'], e[0], e[1], e[2], e[3], e[4])

        result += main + '\n' + entities + '\n'

    return result


def tmtooljson2bern(tmtool_res):
    tmtool_dicts = json.loads(tmtool_res)

    bern_dicts = list()

    for td in tmtool_dicts:
        mutations = list()
        for d in td['denotations']:
            mention = td['text'][d['span']['begin']: d['span']['end']]
            d['span']['end'] += 1
            mutations.append({
                'start': d['span']['start'],
                'end': d['span']['end'],
                'mention': mention,
                'normalizedName': d['obj'].replace('Mutation:', '')
            })

        doc_dict = {
            'pmid': td['sourceid'],
            'text': td['text'],
            'entities': {'mutation': mutations}
        }

        bern_dicts.append(doc_dict)

    return bern_dicts


if __name__ == '__main__':
    # xmlerr, xmlvalid = query_pubtator2biocxml('21660500')
    # print(xmlerr)
    # print(xmlvalid)
    # ElTree.fromstring(xmlvalid)
    # try:
    #     ElTree.fromstring(xmlerr)
    # except ElTree.ParseError as parsee:
    #     print(parsee.msg)
    #
    # dl = pubtator_biocxml2dict_list([21660500])
    # print(dl)

    dl = pubtator_biocxml2dict_list([21581243])
    for doc in dl:
        print(doc)
        print(doc['pmid'])
        print(doc['title'])
        abst = doc['abstract']
        print(abst)
