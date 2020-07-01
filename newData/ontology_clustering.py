def createTwoLevelOntoTable(yago):
    total = {}
    # 第一层ontology，最抽象的一层
    level1 = {}
    # 第二层ontology
    level2 = {}
    # xxx_onto_isa.txt是我把数据中的isa关系涉及的三元组单独取出来建立的文件
    filename = "dbpedia/db_onto_isa.txt"
    if yago:
        filename = "yago/yago_onto_isa.txt"

    file = open(filename, "r")
    lines = file.readlines()
    for line in lines:
        entity1, entity2 = line.split()[0], line.split()[2]
        # 先把自环消除掉
        if entity1 == entity2:
            continue
        if entity1 in level1:
            level1.pop(entity1)
            level2[entity1] = 2
        elif entity1 not in level2:
            level2[entity1] = 2

        if entity2 not in level1 and entity2 not in level2:
            level1[entity2] = 1
    # 还需要建立一个一级ontology与二级ontology的映射表
    while True:
        writeFlag = False
        for item in level2.items():
            for line in lines:
                entity1, entity2 = line.split()[0], line.split()[2]
                if entity1 == item[0]:
                    if entity2 in level1 and level2[entity1] == 2:
                        level2[entity1] = entity2
                        writeFlag = True
                    elif entity2 in level2 and level2[entity2] != 2 and level2[entity1] == 2:
                        level2[entity1] = level2[entity2]
                        writeFlag = True
        if not writeFlag:
            break
    # 在level2中能找到对应的key的，才是二级节点，否则就还是一级
    file.close()
    return level2

def readInsType(yago):
    filename = "dbpedia/db_InsType_mini.txt"
    if yago:
        filename = "yago/yago_InsType_mini.txt"

    file = open(filename, "r")
    print(1)
    map_table = createTwoLevelOntoTable(yago)
    level1 = {}
    level2 = {}
    for line in file.readlines():
        entity, ontology = line.split()[0], line.split()[2]
        if ontology in map_table:
            level1[entity] = map_table[ontology]
            level2[entity] = ontology
        else:
            level1[entity] = ontology
    file.close()
    return level1, level2


# 先处理dbpedia
level1, level2 = readInsType(False)


