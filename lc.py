def largestSubmatrix(matrix: List[List[int]]) -> int:
        height_map = [[0]*len(matrix[0])]*len(matrix)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 1 and i == 0:
                    height_map[i][j] = 1
                elif matrix[i][j] == 1:
                    height_map[i][j] = height_map[i-1][j] + 1
                print(i,j)
                print(height_map[i][j])
        print(height_map)

largestSubmatrix([[0,0,1],[1,1,1],[1,0,1]])