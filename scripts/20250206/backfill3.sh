
STEP=131072
rank=3
START=0+STEP*$rank
END=29097984
# skip_numbers=("0" "262144" "655360" "917504" "1441792" "2490368" "2883584" "3145728")
skip_numbers=("131072" "2490368" "4849664" "5111808" "6291456" "8912896" "9306112" "9830400" "10878976" "11272192" "11927552" "12845056" "15466496" "15990784" "18219008" "21626880" "23068672" "23461888" "23592960" "27525120" "28704768")
for ((i=START; i<=END; i+=STEP*8)); do
    # 检查是否在跳过列表中
    if ! [[ " ${skip_numbers[@]} " =~ " ${i} " ]]; then
        echo "Skipping $i"
        continue
    fi
    echo "$i"
    bash /mnt/nj-larc/usr/ajie1/sunyifei/nlp_codebase/infer.sh $rank $i
done