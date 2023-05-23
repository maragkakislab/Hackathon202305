chris_path = "/home/leect/workspace/slide"

samples = [
        "mmu_dRNA_3T3_mion_1",
        "mmu_dRNA_3T3_PION_1",
        ]

rule run_all:
    input: 
        expand("prep/experiments/{s}/transcript_bin_distribution.tab", s = samples),
        expand("prep/experiments/{s}/joined_features.tab", s=samples)



rule run_extract_bam_features:
    input:
        bam= chris_path + "/samples/{s}/align/reads.sanitize.noribo.toTranscriptome.sorted.uxi.bam",
        bed= chris_path + "/data/mm10/genic_elements.mrna.bed",
    output:
        meta_coords= "{s}.meta_coordinates.tab",
        hist5p= "{s}.hist5p.tab",
        hist3p= "{s}.hist3p.tab",
        pdf= "{s}.pdf",
    shell:
        """
        python scripts/extract_bam_meta_features.py \
                -i {input.bam} \
                -b {input.bed} \
                --bins 10 \
                -n {wildcards.s} \
                --pdf {output.pdf} \
                --ohist_5p {output.hist5p} \
                --ohist_3p {output.hist3p} \
                > {output.meta_coords}
        """

rule aggregate_transcript_meta_features:
    input:
        meta_coords = "{s}.meta_coordinates.tab",
    output:
        transcr_bin_distro = "prep/experiments/{s}/transcript_bin_distribution.tab",
    shell:
        """
        python scripts/aggregate-transcript-meta-features.py \
                --input {input.meta_coords} \
                > {output.transcr_bin_distro}
        """

rule join_features:
    input: 
        transcr_bin_distro = "prep/experiments/{s}/transcript_bin_distribution.tab",
        transc_feature = "prep/experiments/{s}/features_v1.csv"
    params: key1 = "transcript", key2 = "transcript_id"
    output: joined_table = "prep/experiments/{s}/joined_features.tab"
    shell:
        """
        awk 'BEGIN {{ FS = \",\"; OFS = \"\t\" }} {{$1=$1; print }}' {input.transc_feature} | 
            table-join.py -t - \
            -g {input.transcr_bin_distro} \
            -c {params.key1} \
            -d {params.key2} > {output.joined_table}
        """


