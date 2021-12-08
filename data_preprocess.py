
with open('UnderlyingClusters.txt') as reader, open('training_data.txt', 'w') as writer:
    for index, line in enumerate(reader):
        if index%11 != 0:
            writer.write(line)

with open("training_data.txt", "r") as my_train_file, open("EncodedStrands.txt", "r") as my_train_file_2:
    data_1 = my_train_file.read().splitlines()
    data_2 = my_train_file.read().splitlines()

with open("predata.txt", "w") as file:

    for i in range(len(data_1)):
        data_1[i] = list(data_1[i])

    file = open("predata.txt", "w")

    for i in range(int(len(data_1)/10)):
        data_concat = []
        max_len = 0

        for j in range(10):
            max_len = max(max_len, len(data_1[i * 10 + j]))
        for j in range(120):
            print("i: ", i, j)
            data_concat_2 = []
            for k in range(10):
                try:
                    data_concat_2.append(data_1[10 * i + k][j])
                except:
                    data_concat_2.append(str(1))
            data_concat.append("".join(data_concat_2))

            file.writelines("".join(data_concat_2))
        file.writelines(",")
file.close()

