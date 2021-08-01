import numpy as np

def create_labels(df,col_name,window_size=11):
    row_counter = 0
    total_rows = len(df)
    labels = np.zeros(total_rows)
    labels[:]=np.nan
    while row_counter <total_rows:
        if row_counter>=window_size-1:
            window_begin=row_counter-(window_size-1)
            window_end=row_counter
            window_middle=(window_begin+window_end)/2
            min_=np.inf
            min_index=-1
            max_=-np.inf
            max_index=-1
            for i in range(window_begin,window_end+1):
                price = df.iloc[i][col_name]
                if price<min_:
                    min_=price
                    min_index=i
                elif price>max_:
                    max_=price
                    max_index=i
            if max_index == window_middle:
                labels[row_counter] = 2
            elif min_index == window_middle:
                labels[row_counter] = 0
            else:
                labels[row_counter] = 1
        row_counter = row_counter+1
    return labels