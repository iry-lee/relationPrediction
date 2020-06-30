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

# createValidFile(True)
# createValidFile(False)

# createID(True)
# createID(False)

randomInitialVec(True)
randomInitialVec(False)
