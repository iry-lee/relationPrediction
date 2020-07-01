import random

# 创建验证集
# 看FB15K-237中验证集中的数据数量大概是与测试集的数量相似
def createValidFile(yago):
    folder = "dbpedia"
    sub_path = "/db"
    train_lines = 592654
    valid_lines = 65851
    if yago:
        folder = "yago"
        sub_path = "/yago"
        train_lines = 351664
        valid_lines = 39074
    f_train = open(folder + sub_path + "_insnet_train.txt", "r")
    f_valid = open(folder + "/valid.txt", "w")
    f_train_new = open(folder + "/train.txt", "w")
    valid_list = random.sample(range(0, train_lines), valid_lines)

    lines = f_train.readlines()
    for i in valid_list:
        f_valid.write(lines[i])
    for i in range(0, train_lines):
        if i not in valid_list:
            f_train_new.write(lines[i])

    f_train.close()
    f_valid.close()
    f_train_new.close()

# 创建entity2id.txt和relation2id.txt
def createID(yago):
    filename = "dbpedia/db_insnet.txt"
    entity2id = "dbpedia/entity2id.txt"
    relation2id = "dbpedia/relation2id.txt"
    if yago:
        filename = "yago/yago_insnet_mini.txt"
        entity2id = "yago/entity2id.txt"
        relation2id = "yago/relation2id.txt"

    input_file = open(filename, "r")
    entity_file = open(entity2id, "w")
    relation_file = open(relation2id, "w")

    entity_table = {}
    relation_table = {}
    entity_table_ct = 0
    relation_table_ct = 0
    for line in input_file.readlines():
        entity1, relation, entity2 = line.split()[0], line.split()[1], line.split()[2]
        if entity1 not in entity_table:
            entity_table[entity1] = entity_table_ct
            entity_table_ct = entity_table_ct + 1
        if entity2 not in entity_table:
            entity_table[entity2] = entity_table_ct
            entity_table_ct = entity_table_ct + 1
        if relation not in relation_table:
            relation_table[relation] = relation_table_ct
            relation_table_ct = relation_table_ct + 1

    for item in entity_table.items():
        entity_file.write(item[0] + '\t' + str(item[1]) + '\n')

    for item in relation_table.items():
        relation_file.write(item[0] + '\t' + str(item[1]) + '\n')

    input_file.close()
    entity_file.close()
    relation_file.close()

# 对实体和关系向量进行随机初始化
def randomInitialVec(yago):
    dimension = 100  # 向量的维度
    n_entity = 98336
    n_relation = 298
    entity2vec = "dbpedia/entity2vec.txt"
    relation2vec = "dbpedia/relation2vec.txt"
    if yago:
        n_entity = 26078
        n_relation = 34
        entity2vec = "yago/entity2vec.txt"
        relation2vec = "yago/relation2vec.txt"

    entity2vec_file = open(entity2vec, 'w')
    relation2vec_file = open(relation2vec, "w")
    for i in range(n_entity):
        for j in range(dimension):
            r = random.random()
            entity2vec_file.write(str(r)[:8] + "\t")
        entity2vec_file.write("\n")
    entity2vec_file.close()

    for i in range(n_relation):
        for j in range(dimension):
            r = random.random()
            relation2vec_file.write(str(r)[:8] + "\t")
        relation2vec_file.write("\n")
    relation2vec_file.close()

# 使用ontology对entity进行clustering
# 想一下如何做这个聚类，如何根据层级设置margin，怎么确定这个类是高层次还是低层次？通过isa？这里可以跟师兄讨论一下
def findLevelofISA(yago):
    filename = "dbpedia/db_onto_isa.txt"
    if yago:
        filename = "yago/yago_onto_isa.txt"

    file = open(filename, "r")
    table = {}
    lines = file.readlines()

    # 只执行一遍for循环，可能会存在消息延迟的问题，得不到最终的层级
    # 但因为isa的关系可能存在闭环，所以下面while循环可能会死循环
    ct = 0
    while True:
        flag = False
        ct = ct + 1
        for i in range(0, len(lines)):
            head_onto = lines[i].split()[0]
            tail_onto = lines[i].split()[2]
            # 暂且消除自环
            if tail_onto == head_onto:
                continue
            if head_onto not in table:
                table[head_onto] = 1
                flag = True
            if tail_onto not in table:
                table[tail_onto] = 1
                flag = True

            if table[tail_onto] < table[head_onto] + 1:
                table[tail_onto] = table[head_onto] + 1
                flag = True
        if not flag or ct > 100:
            break

    for item in table.items():
        print(item)


# createValidFile(True)
# createValidFile(False)

# createID(True)
# createID(False)

# randomInitialVec(True)
# randomInitialVec(False)
