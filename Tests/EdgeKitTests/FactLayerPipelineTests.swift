// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class FactLayerPipelineTests: XCTestCase {

    var tmpDir: URL!
    var store: FactStore!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("pipeline-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        store = try FactStore(path: tmpDir.appendingPathComponent("data.sqlite3"))
    }

    override func tearDownWithError() throws {
        store?.close()
        try? FileManager.default.removeItem(at: tmpDir)
    }

    func test_defaultPipelineFailsClosedWithoutRulePlanning() throws {
        let pipeline = FactLayerPipeline(store: store)

        let result = try pipeline.chat(query: "上月花了多少钱")

        XCTAssertEqual(result.routerResult.label, .baseChat)
        XCTAssertEqual(result.routerResult.reason, "legacy_rule_based_fallback_disabled")
        XCTAssertNil(result.answerHint)
        XCTAssertTrue(result.factsUsed.isEmpty)
        XCTAssertEqual(result.prompt.injectedFactCount, 0)
        XCTAssertFalse(result.prompt.system.contains(FACT_START))
    }

    func test_learnedFactRouteDoesNotRunRulePlanningOrFactLookup() throws {
        let pipeline = FactLayerPipeline(
            store: store,
            router: QueryRouter(base: MockBaseClassifier(
                defaultLabel: .fact,
                defaultConfidence: 0.95
            ))
        )

        let result = try pipeline.chat(query: "上月花了多少钱")

        XCTAssertEqual(result.routerResult.label, .fact)
        XCTAssertEqual(result.routerResult.reason, "few_shot_high_conf")
        XCTAssertNil(result.answerHint)
        XCTAssertTrue(result.factsUsed.isEmpty)
        XCTAssertEqual(result.prompt.injectedFactCount, 0)
        XCTAssertFalse(result.prompt.system.contains(FACT_START))
    }

    func test_learnedMixedRouteDoesNotRunRulePlanningOrFactLookup() throws {
        let pipeline = FactLayerPipeline(
            store: store,
            router: QueryRouter(base: MockBaseClassifier(
                defaultLabel: .mixed,
                defaultConfidence: 0.95
            ))
        )

        let result = try pipeline.chat(query: "根据我的历史推荐")

        XCTAssertEqual(result.routerResult.label, .mixed)
        XCTAssertNil(result.answerHint)
        XCTAssertTrue(result.factsUsed.isEmpty)
        XCTAssertEqual(result.prompt.injectedFactCount, 0)
    }

    func test_personaRouteStillSuppressesFactInjection() throws {
        let pipeline = FactLayerPipeline(
            store: store,
            router: QueryRouter(base: MockBaseClassifier(
                defaultLabel: .persona,
                defaultConfidence: 0.95
            ))
        )

        let result = try pipeline.chat(query: "我是什么样的人")

        XCTAssertEqual(result.routerResult.label, .persona)
        XCTAssertNil(result.answerHint)
        XCTAssertTrue(result.factsUsed.isEmpty)
        XCTAssertEqual(result.prompt.injectedFactCount, 0)
        XCTAssertFalse(result.prompt.system.contains(FACT_START))
    }

    func test_configKeepsSourceCompatibleFieldsWithoutRulePlanning() {
        let refDate = Date(timeIntervalSince1970: 1_777_777_777)
        let config = FactLayerPipeline.Config(
            referenceDate: refDate,
            listLimit: 7,
            tokenBudget: 128,
            systemPromptPrefix: "system",
            injectionMode: .inlineUser
        )

        XCTAssertEqual(config.referenceDate, refDate)
        XCTAssertEqual(config.listLimit, 7)
        XCTAssertEqual(config.tokenBudget, 128)
        XCTAssertEqual(config.systemPromptPrefix, "system")
        XCTAssertEqual(config.injectionMode, .inlineUser)
    }
}
