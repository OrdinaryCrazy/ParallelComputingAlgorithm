void merge(int* arraySource, int len1, int len2, int* arrayDest)
{
    int index1 = 0, index2 = len1;
    for(int i = 0; i < len1 + len2; i++)
    {
        if(index1 == len1)
        {
            arrayDest[i] = arraySource[index2++];
        }
        else
        {
            if(index2 == len1 + len2)
            {
                arrayDest[i] = arraySource[index1++];
            }
            else
            {
                if(arraySource[index1] > arraySource[index2])   arrayDest[i] = arraySource[index2++];
                else                                            arrayDest[i] = arraySource[index1++];
            }
        }
    }
}
void multiMergeSort(int* arraySource, int* div, int* arrayDest, int groupSize)
{
    int j = 0;
    for(int i = 0; i < groupSize; i++)
    {
        if(div[i] > 0)
        {
            div[j++] = div[i];
            if(j < i + 1) div[i] = 0;
        }
    }
    if(j > 1)
    {
        int n = 0;
        for(int i = 0; i + 1 < j; i++)
        {
            merge(&arraySource[n], div[i], div[i + 1], &arrayDest[n]);
            div[i] += div[i + 1];
            div[i + 1] = 0;
            n += div[i];
        }
        if(j % 2 == 1)
        {
            for(int i = 0; i < div[j - 1]; i++, n++)
            {
                arrayDest[n] = arraySource[n];
            }
        }
        multiMergeSort(arrayDest, div, arraySource, groupSize);
    }
}
